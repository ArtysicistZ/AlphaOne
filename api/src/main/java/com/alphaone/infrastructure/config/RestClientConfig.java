package com.alphaone.infrastructure.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.lang.NonNull;
import org.springframework.web.client.RestClient;
import org.springframework.beans.factory.annotation.Value;

@Configuration
public class RestClientConfig {

    @Bean
    public RestClient inferenceRestClient(
        @NonNull @Value("${inference.api.base-url}") String baseUrl
    ) {
        return RestClient.builder()
            .baseUrl(baseUrl)
            .build();
    }
}
