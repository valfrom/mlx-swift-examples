// Copyright © 2025 Apple Inc.

import MLX
import MLXLLM
import MLXRandom
import XCTest

final class TransformersLlamaModelTests: XCTestCase {

    func testLeftPaddedPromptMatchesHuggingFaceLogits() {
        MLXRandom.seed(42)

        let config = LlamaConfiguration(
            hiddenSize: 32,
            hiddenLayers: 2,
            intermediateSize: 64,
            attentionHeads: 4,
            headDimensions: 8,
            rmsNormEps: 1e-5,
            vocabularySize: 16,
            kvHeads: 4,
            maxPositionEmbeddings: 2048,
            ropeTheta: 10_000,
            ropeTraditional: false,
            ropeScaling: nil,
            tieWordEmbeddings: true,
            attentionBias: false,
            mlpBias: false
        )

        let model = TransformersLlamaModel(config)

        let unpaddedInputIds = MLXArray([[Int32(1), 2, 3]])
        let unpaddedMask = MLXArray([[Int32(1), 1, 1]])
        let unpaddedPositionIds = MLXArray([[Int32(10), 11, 12]])

        let paddedInputIds = MLXArray([[Int32(0), 1, 2, 3]])
        let paddedMask = MLXArray([[Int32(0), 1, 1, 1]])
        let paddedPositionIds = MLXArray([[Int32(0), 10, 11, 12]])

        let unpaddedOutput = model.forward(
            inputIds: unpaddedInputIds,
            attentionMask: unpaddedMask,
            positionIds: unpaddedPositionIds,
            useCache: false
        )

        let paddedOutput = model.forward(
            inputIds: paddedInputIds,
            attentionMask: paddedMask,
            positionIds: paddedPositionIds,
            useCache: false
        )

        let embedding = model.getInputEmbeddings()
        let unpaddedLogits = embedding.asLinear(unpaddedOutput.lastHiddenState)
        let paddedLogits = embedding.asLinear(paddedOutput.lastHiddenState)

        // Treat the unpadded run as the Hugging Face baseline since HF also
        // computes position embeddings from the non-padded positions.
        let referenceEOSLogits = unpaddedLogits[0, 2, 0...].asArray(Float.self)
        let paddedEOSLogits = paddedLogits[0, 3, 0...].asArray(Float.self)

        XCTAssertEqual(referenceEOSLogits.count, paddedEOSLogits.count)

        for (reference, padded) in zip(referenceEOSLogits, paddedEOSLogits) {
            XCTAssertEqual(reference, padded, accuracy: 1e-4)
        }
    }
}
