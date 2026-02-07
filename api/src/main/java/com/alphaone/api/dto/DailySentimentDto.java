package com.alphaone.api.dto;

import java.time.LocalDate;

public record DailySentimentDto(LocalDate date, Double averageScore) {}
