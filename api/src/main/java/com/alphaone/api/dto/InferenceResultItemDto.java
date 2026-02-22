package com.alphaone.api.dto;

public record InferenceResultItemDto(
    String sentence,
    String normalizedInput,
    String target,
    String sentiment,
    double score
) {}
