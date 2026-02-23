package com.alphaone.api.dto;

import java.util.List;

public record AttentionDataDto(
    List<String> tokens,
    List<List<Double>> matrix
) {}
