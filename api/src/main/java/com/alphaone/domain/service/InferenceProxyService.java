package com.alphaone.domain.service;

import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.alphaone.api.dto.InferenceRequestDto;
import com.alphaone.api.dto.InferenceResponseDto;

@Service
public class InferenceProxyService {
    
    private static final Logger logger = LoggerFactory.getLogger(InferenceProxyService.class);
    private final RestClient inferenceRestClient;

    public InferenceProxyService(RestClient inferenceRestClient) {
        this.inferenceRestClient = inferenceRestClient;
    }

    public InferenceResponseDto runInference(InferenceRequestDto request) {
        logger.info(
            "inference.proxy.request text_len={} targets={}",
            request.text().length(), request.targets()
        );
        InferenceResponseDto response = inferenceRestClient.post()
            .uri("/api/v1/inference")
            .body(request)
            .retrieve()
            .body(InferenceResponseDto.class);

        logger.info(
            "inference.proxy.response result={}", 
            response != null ? response.results().size() : 0
        );
        return response;
    }

}
