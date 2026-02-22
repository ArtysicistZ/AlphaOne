package com.alphaone.api.dto;

import java.util.List;

public record InferenceResponseDto(
    List<InferenceResultItemDto> results
) {}
