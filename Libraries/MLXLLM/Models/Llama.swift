// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/llama.py

func computeBaseFrequency(
    base: Float, dims: Int, ropeType: String, ropeScaling: [String: StringOrNumber]?
)
    -> Float
{
    if ropeType != "llama3" {
        return base
    }

    guard let ropeScaling = ropeScaling else {
        return base
    }

    guard case .float(let factor) = ropeScaling["factor"],
        case .float(let lowFreqFactor) = ropeScaling["low_freq_factor"] ?? .float(1.0),
        case .float(let highFreqFactor) = ropeScaling["high_freq_factor"] ?? .float(4.0),
        case .float(let oldContextLen) = ropeScaling["original_max_position_embeddings"]
            ?? .float(8192)
    else {
        return base
    }

    let lowFreqWavelen = oldContextLen / lowFreqFactor
    let highFreqWavelen = oldContextLen / highFreqFactor

    let freqs = (0 ..< dims).compactMap { index -> Float? in
        if index % 2 == 0 {
            return pow(base, Float(index) / Float(dims))
        }
        return nil
    }

    let newBaseFreqs = freqs.map { freq -> Float in
        let wavelen = 2 * .pi / freq
        let smooth = max(
            0, min(1, (wavelen - highFreqWavelen) / (lowFreqWavelen - highFreqWavelen)))
        return freq * ((1 - smooth) * factor + smooth)
    }

    return newBaseFreqs.reduce(0, +) / Float(newBaseFreqs.count)
}

private class DynamicNTKScalingRoPE: Module {
    let dims: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    var base: Float?
    let scale: Float
    let ropeType: String
    let ropeScaling: [String: StringOrNumber]?
    var freqs: MLXArray?

    init(
        dims: Int,
        maxPositionEmbeddings: Int?,
        traditional: Bool = false,
        base: Float = 10000,
        scale: Float = 1.0,
        ropeType: String = "default",
        ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings ?? 2048
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.ropeType = ropeType
        self.ropeScaling = ropeScaling
        super.init()
        computeFreqs()
    }

    private func computeFreqs() {
        if ropeType != "llama3" {
            freqs = nil
            return
        }

        guard let ropeScaling = ropeScaling,
            case .float(let factor) = ropeScaling["factor"],
            case .float(let lowFreqFactor) = ropeScaling["low_freq_factor"] ?? .float(1.0),
            case .float(let highFreqFactor) = ropeScaling["high_freq_factor"] ?? .float(4.0),
            case .float(let oldContextLen) = ropeScaling["original_max_position_embeddings"]
                ?? .float(8192),
            let base
        else {
            freqs = nil
            return
        }

        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        let indices = MLXArray(stride(from: 0, to: dims, by: 2))
        var frequencies = MLX.pow(base, indices / Float(dims))
        let wavelens = 2 * Float.pi * frequencies

        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen), frequencies * factor, frequencies)
        let isMediumFreq = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )
        let smoothFactors =
            (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)

        freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
        self.base = nil
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: base,
            scale: scale,
            offset: offset,
            freqs: freqs
        )
    }
}

private class Attention: Module {

    let args: LlamaConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: DynamicNTKScalingRoPE

    init(_ args: LlamaConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        self.rope = DynamicNTKScalingRoPE(
            dims: headDim,
            maxPositionEmbeddings: args.maxPositionEmbeddings,
            traditional: args.ropeTraditional,
            base: args.ropeTheta,
            scale: 1.0,
            ropeType: {
                if case .string(let value) = args.ropeScaling?["type"] {
                    return value
                } else {
                    return "default"
                }
            }(),
            ropeScaling: args.ropeScaling)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let positionEmbeddings {
            queries = applyRotary(
                queries,
                cos: positionEmbeddings.cos,
                sin: positionEmbeddings.sin
            )
            keys = applyRotary(
                keys,
                cos: positionEmbeddings.cos,
                sin: positionEmbeddings.sin
            )
        } else if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }

    private func applyRotary(
        _ tensor: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> MLXArray {
        var cos = cos
        var sin = sin
        if cos.dtype != tensor.dtype {
            cos = cos.asType(tensor.dtype)
        }
        if sin.dtype != tensor.dtype {
            sin = sin.asType(tensor.dtype)
        }

        cos = cos[0..., .newAxis, 0..., 0...]
        sin = sin[0..., .newAxis, 0..., 0...]

        let lastDim = tensor.dim(tensor.ndim - 1)
        let halfDim = lastDim / 2
        let firstHalf = tensor[.ellipsis, ..<halfDim]
        let secondHalf = tensor[.ellipsis, halfDim...]
        let negSecondHalf = secondHalf * MLXArray(-1, dtype: tensor.dtype)
        let rotated = MLX.concatenate([negSecondHalf, firstHalf], axis: -1)

        return tensor * cos + rotated * sin
    }
}

private class MLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: LlamaConfiguration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activation = silu(gate(x))
        return down(activation * up(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "mlp") var mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: LlamaConfiguration) {
        self._attention.wrappedValue = Attention(args)
        self._mlp.wrappedValue = MLP(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> MLXArray {
        var r = attention(
            inputLayerNorm(x),
            mask: mask,
            cache: cache,
            positionEmbeddings: positionEmbeddings
        )
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

private class LlamaModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [TransformerBlock]
    let norm: RMSNorm

    init(_ args: LlamaConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { _ in TransformerBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

/// Model for Llama and Mistral model types.
public class LlamaModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: LlamaModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LlamaConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = LlamaModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Remove unused precomputed rotary frequencies
        weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        // some models allow the system role and some do not -- this is enforced
        // by the chat template (code).
        do {
            let probe = [
                [
                    "role": "system",
                    "content": "test",
                ]
            ]
            _ = try tokenizer.applyChatTemplate(messages: probe)
            return DefaultMessageGenerator()
        } catch {
            return NoSystemMessageGenerator()
        }
    }
}

public struct LlamaModelForwardOutput: Sendable {
    public let lastHiddenState: MLXArray
    public let pastKeyValues: [KVCache]?
    public let hiddenStates: [MLXArray]?
    public let attentions: [MLXArray]?

    public init(
        lastHiddenState: MLXArray,
        pastKeyValues: [KVCache]?,
        hiddenStates: [MLXArray]?,
        attentions: [MLXArray]?
    ) {
        self.lastHiddenState = lastHiddenState
        self.pastKeyValues = pastKeyValues
        self.hiddenStates = hiddenStates
        self.attentions = attentions
    }
}

/// Transformers-aligned Llama model implementation focused on inference.
///
/// Mirrors the high level structure from HuggingFace's ``LlamaModel`` but uses
/// Swift MLX layers and removes training-specific behaviour such as gradient
/// checkpointing.
public final class TransformersLlamaModel: Module {

    public let config: LlamaConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: TransformersRotaryEmbedding

    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm

    public init(_ config: LlamaConfiguration) {
        precondition(config.vocabularySize > 0)
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._rotaryEmbedding.wrappedValue = TransformersRotaryEmbedding(config: config)
        self.layers = (0 ..< config.hiddenLayers).map { _ in TransformerBlock(config) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func getInputEmbeddings() -> Embedding { embedTokens }

    public func setInputEmbeddings(_ value: Embedding) {
        self._embedTokens.wrappedValue = value
    }

    public func forward(
        inputIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        positionIds: MLXArray? = nil,
        pastKeyValues: [KVCache]? = nil,
        inputsEmbeds: MLXArray? = nil,
        useCache: Bool? = nil,
        outputAttentions: Bool? = nil,
        outputHiddenStates: Bool? = nil,
        returnDict: Bool? = nil,
        cachePosition: MLXArray? = nil
    ) -> LlamaModelForwardOutput {
        precondition(
            (inputIds != nil) != (inputsEmbeds != nil),
            "You must specify exactly one of inputIds or inputsEmbeds")

        if let returnDict, !returnDict {
            fatalError("Tuple outputs are not supported in TransformersLlamaModel")
        }

        if outputAttentions ?? false {
            fatalError("outputAttentions is not supported in TransformersLlamaModel")
        }

        let useCache = useCache ?? (config.hiddenLayers > 0)
        let collectHiddenStates = outputHiddenStates ?? false

        var hiddenStates: MLXArray
        if let inputsEmbeds {
            hiddenStates = inputsEmbeds
        } else if let inputIds {
            hiddenStates = embedTokens(inputIds)
        } else {
            fatalError("Invalid inputs for TransformersLlamaModel")
        }

        var cache = useCache ? pastKeyValues : nil
        if useCache && cache == nil {
            cache = (0 ..< layers.count).map { _ in KVCacheSimple() }
        }

        let (maskMode, _, positionIdsResolved) = prepareMaskAndPositions(
            attentionMask: attentionMask,
            hiddenStates: hiddenStates,
            cache: cache,
            cachePosition: cachePosition,
            positionIds: positionIds
        )

        var allHiddenStates: [MLXArray] = []
        if collectHiddenStates {
            allHiddenStates.reserveCapacity(layers.count + 1)
        }

        let positionEmbeddings = rotaryEmbedding(
            hiddenStates: hiddenStates,
            positionIds: positionIdsResolved
        )

        for (idx, layer) in layers.enumerated() {
            if collectHiddenStates {
                allHiddenStates.append(hiddenStates)
            }

            hiddenStates = layer(
                hiddenStates,
                mask: maskMode,
                cache: cache?[idx],
                positionEmbeddings: positionEmbeddings
            )
        }

        hiddenStates = norm(hiddenStates)

        if collectHiddenStates {
            allHiddenStates.append(hiddenStates)
        }

        let nextCache = useCache ? cache : nil

        return LlamaModelForwardOutput(
            lastHiddenState: hiddenStates,
            pastKeyValues: nextCache,
            hiddenStates: collectHiddenStates ? allHiddenStates : nil,
            attentions: nil
        )
    }

    private final class TransformersRotaryEmbedding: Module {

        let dims: Int
        let base: Float
        let ropeScaling: [String: StringOrNumber]?
        let ropeType: String
        let maxPositionEmbeddings: Int

        @ModuleInfo(key: "inv_freq") var inverseFrequencies: MLXArray

        init(config: LlamaConfiguration) {
            self.dims = config.resolvedHeadDimensions
            self.maxPositionEmbeddings = config.maxPositionEmbeddings ?? 2048
            self.ropeScaling = config.ropeScaling
            if let ropeScaling,
                let rawType = ropeScaling["type"] ?? ropeScaling["rope_type"],
                case .string(let resolvedType) = rawType
            {
                self.ropeType = resolvedType
            } else {
                self.ropeType = "default"
            }

            let resolvedBase = computeBaseFrequency(
                base: config.ropeTheta,
                dims: dims,
                ropeType: ropeType,
                ropeScaling: config.ropeScaling
            )
            self.base = resolvedBase

            let exponent = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32) / Float(dims)
            let invFreq = MLXArray(1.0) / MLX.pow(MLXArray(resolvedBase), exponent)
            self._inverseFrequencies.wrappedValue = invFreq
        }

        private func applyScalingIfNeeded(_ positionIds: MLXArray) -> MLXArray {
            guard let ropeScaling,
                case .string(let type) = ropeScaling["type"] ?? ropeScaling["rope_type"],
                case .float(let factor) = ropeScaling["factor"]
            else {
                return positionIds
            }

            switch type {
            case "linear", "dynamic":
                return positionIds / MLXArray(factor)
            default:
                return positionIds
            }
        }

        func callAsFunction(
            hiddenStates: MLXArray,
            positionIds: MLXArray
        ) -> (cos: MLXArray, sin: MLXArray) {
            var positionIds = positionIds
            if positionIds.dtype != .float32 {
                positionIds = positionIds.asType(.float32)
            }
            positionIds = applyScalingIfNeeded(positionIds)

            var invFreq = inverseFrequencies
            if invFreq.dtype != .float32 {
                invFreq = invFreq.asType(.float32)
            }

            var freqs = positionIds[.ellipsis, .newAxis]
            freqs = freqs * invFreq[.newAxis, .newAxis, 0...]
            freqs = MLX.concatenate([freqs, freqs], axis: -1)

            var cos = MLX.cos(freqs)
            var sin = MLX.sin(freqs)

            let targetType = hiddenStates.dtype
            if cos.dtype != targetType {
                cos = cos.asType(targetType)
            }
            if sin.dtype != targetType {
                sin = sin.asType(targetType)
            }

            return (cos, sin)
        }
    }

    private func prepareMaskAndPositions(
        attentionMask: MLXArray?,
        hiddenStates: MLXArray,
        cache: [KVCache]?,
        cachePosition: MLXArray?,
        positionIds: MLXArray?
    ) -> (MLXFast.ScaledDotProductAttentionMaskMode, MLXArray, MLXArray) {
        let sequenceLength = hiddenStates.dim(1)
        let batchSize = hiddenStates.dim(0)

        let pastSeenTokens = cache?.first?.offset ?? 0
        let resolvedCachePosition: MLXArray
        if let cachePosition {
            resolvedCachePosition = cachePosition
        } else {
            resolvedCachePosition = MLXArray(
                Int32(pastSeenTokens) ..< Int32(pastSeenTokens + sequenceLength))
        }

        let resolvedPositionIds: MLXArray
        if let positionIds {
            resolvedPositionIds = positionIds
        } else {
            resolvedPositionIds = resolvedCachePosition[.newAxis, 0...]
        }

        let mask = buildAttentionMask(
            attentionMask: attentionMask,
            sequenceLength: sequenceLength,
            batchSize: batchSize,
            cachePosition: resolvedCachePosition,
            hiddenStates: hiddenStates,
            cache: cache
        )

        return (mask, resolvedCachePosition, resolvedPositionIds)
    }

    private func buildAttentionMask(
        attentionMask: MLXArray?,
        sequenceLength: Int,
        batchSize: Int,
        cachePosition: MLXArray,
        hiddenStates: MLXArray,
        cache: [KVCache]?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if sequenceLength <= 1 && attentionMask == nil {
            return .none
        }

        if attentionMask == nil,
            cache?.first?.maxSize == nil,
            sequenceLength > 1
        {
            return .causal
        }

        let dtype = hiddenStates.dtype
        let pastSeenTokens = cache?.first?.offset ?? 0
        let targetLength: Int
        if let maxSize = cache?.first?.maxSize {
            targetLength = maxSize
        } else if let attentionMask {
            targetLength = attentionMask.dim(attentionMask.ndim - 1)
        } else {
            targetLength = pastSeenTokens + sequenceLength
        }

        let negLarge = MLXArray(-Float(1e9), dtype: dtype)
        let queryPositions = cachePosition.asType(.int32)[0..., .newAxis]
        let keyPositions = MLXArray(Int32(0) ..< Int32(targetLength))[.newAxis, 0...]
        var causalMask2D = (keyPositions .> queryPositions).asType(dtype) * negLarge
        var mask = causalMask2D[.newAxis, .newAxis, 0..., 0...]

        if let attentionMask {
            let maskLength = attentionMask.dim(attentionMask.ndim - 1)
            var paddingMask = MLXArray.zeros([attentionMask.dim(0), targetLength], dtype: dtype)
            let ones = MLXArray.ones(attentionMask.shape, dtype: dtype)
            let adjustment = (attentionMask.asType(dtype) - ones) * MLXArray(Float(1e9), dtype: dtype)
            paddingMask[0..., ..<maskLength] = adjustment
            mask = mask + paddingMask[0..., .newAxis, .newAxis, 0...]
        }

        if batchSize > 1 {
            mask = mask + MLXArray.zeros([batchSize, 1, 1, 1], dtype: dtype)
        }

        return .array(mask)
    }
}

public struct LlamaConfiguration: Codable, Sendable {

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool = true
    var attentionBias: Bool = false
    var mlpBias: Bool = false

    public init(
        hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int, attentionHeads: Int,
        headDimensions: Int? = nil, rmsNormEps: Float, vocabularySize: Int, kvHeads: Int,
        maxPositionEmbeddings: Int? = nil, ropeTheta: Float = 10_000, ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil, tieWordEmbeddings: Bool = true,
        attentionBias: Bool = false, mlpBias: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
    }

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        if let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            self.ropeTheta = ropeTheta
        }
        if let ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
        {
            self.ropeTraditional = ropeTraditional
        }
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        if let tieWordEmbeddings = try container.decodeIfPresent(
            Bool.self, forKey: .tieWordEmbeddings)
        {
            self.tieWordEmbeddings = tieWordEmbeddings
        }
        if let attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) {
            self.attentionBias = attentionBias
        }
        if let mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) {
            self.mlpBias = mlpBias
        }

        if let ropeScaling {
            if ropeScaling["factor"] == nil {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling, in: container,
                    debugDescription: "rope_scaling must contain 'factor'")
            }
            if let ropeType = ropeScaling["type"] ?? ropeScaling["rope_type"] {
                if case .string = ropeType {
                    let options = [
                        StringOrNumber.string("linear"), StringOrNumber.string("dynamic"),
                        StringOrNumber.string("llama3"),
                    ]
                    if !options.contains(ropeType) {
                        throw DecodingError.dataCorruptedError(
                            forKey: .ropeScaling, in: container,
                            debugDescription:
                                "rope_scaling 'type' currently only supports 'linear', 'dynamic', or 'llama3'"
                        )
                    }
                }
            } else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling, in: container,
                    debugDescription: "rope_scaling must contain either 'type' or 'rope_type'")
            }
        }
    }
}

// MARK: - LoRA

extension LlamaModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
