package com.alphaone.api.controller;

import com.alphaone.api.dto.InferenceRequestDto;
import com.alphaone.api.dto.InferenceResponseDto;

import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.alphaone.domain.service.InferenceProxyService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.PostMapping;
import jakarta.validation.Valid;

@RestController
@RequestMapping("/api/v1")
public class InferenceController {
    
    private final InferenceProxyService inferenceProxyService;
    private static final Logger logger = LoggerFactory.getLogger(InferenceController.class);

    public InferenceController(InferenceProxyService inferenceProxyService) {
        this.inferenceProxyService = inferenceProxyService;
    }

    @PostMapping("/inference")
    public InferenceResponseDto runInference(
        @Valid 
        @RequestBody 
        InferenceRequestDto request
    ) {
        logger.info("api.inference.request");
        return inferenceProxyService.runInference(request);
    }
    
}
