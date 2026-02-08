package com.alphaone.infrastructure.logging;

import java.io.IOException;
import java.util.UUID;

import org.springframework.web.filter.OncePerRequestFilter;
import org.springframework.stereotype.Component;
import org.springframework.lang.NonNull;
import org.slf4j.MDC;

import jakarta.servlet.FilterChain;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.ServletException;

@Component
public class RequestIdFilter extends OncePerRequestFilter {

    private static final String REQUEST_ID_HEADER = "X-Request-ID";
    private static final String REQUEST_ID = "requestId";

    @Override
    protected void doFilterInternal(
        @NonNull HttpServletRequest request,
        @NonNull HttpServletResponse response,
        @NonNull FilterChain filterChain
    ) throws IOException, ServletException {
        String requestId = request.getHeader(REQUEST_ID_HEADER);
        
        if (requestId == null || requestId.isEmpty()) {
            requestId = UUID.randomUUID().toString();
        }

        MDC.put(REQUEST_ID, requestId);
        response.setHeader(REQUEST_ID_HEADER, requestId);

        try {
            filterChain.doFilter(request, response);
        } finally {
            MDC.remove("requestId");
        }
    }
    
}
