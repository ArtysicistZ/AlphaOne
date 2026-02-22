package com.alphaone.api.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;
import java.util.List;

public record InferenceRequestDto(
    @NotBlank @Size(max = 1000) String text,
    @NotEmpty List<String> targets
) {}
