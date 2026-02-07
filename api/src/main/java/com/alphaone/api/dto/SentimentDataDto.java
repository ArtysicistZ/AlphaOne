package com.alphaone.api.dto;

import java.time.OffsetDateTime;

public record SentimentDataDto(
    Long id, 
    OffsetDateTime createdAt,
    String sentimentLabel,
    Double sentimentScore,
    String relevantText
) {}
