package com.alphaone.api.controller;

import com.alphaone.api.dto.ErrorResponse;

import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jakarta.servlet.http.HttpServletRequest;

import java.time.OffsetDateTime;


@RestController
public class ApiExceptionHandler {

    private static final Logger logger = LoggerFactory.getLogger(ApiExceptionHandler.class);
    
    @ExceptionHandler(ResponseStatusException.class)
    public ResponseEntity<ErrorResponse> handleResponseStatus(
        ResponseStatusException ex,
        HttpServletRequest request
    ) {
        HttpStatus status = HttpStatus.valueOf(ex.getStatusCode().value());
        ErrorResponse errorResponse = new ErrorResponse(
            OffsetDateTime.now(),
            status.value(),
            status.getReasonPhrase(),
            ex.getReason(),
            request.getRequestURI()
        );
        logger.warn(
            "api_exception type=ResponseStatusException status={} reason={} path={}",
            status.value(),
            ex.getReason(),
            request.getRequestURI()
        );
        return ResponseEntity.status(status).body(errorResponse);
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleMethodArgumentNotValid(
        MethodArgumentNotValidException ex,
        HttpServletRequest request
    ) {
        HttpStatus status = HttpStatus.BAD_REQUEST;
        ErrorResponse errorResponse = new ErrorResponse(
            OffsetDateTime.now(),
            status.value(),
            status.getReasonPhrase(),
            "Validation failed",
            request.getRequestURI()
        );
        logger.warn(
            "api_exception type=MethodArgumentNotValidException status={} message={} path={}",
            status.value(),
            ex.getMessage(),
            request.getRequestURI()
        );
        return ResponseEntity.status(status).body(errorResponse);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGenericException(
        Exception ex,
        HttpServletRequest request
    ) {
        HttpStatus status = HttpStatus.INTERNAL_SERVER_ERROR;
        ErrorResponse errorResponse = new ErrorResponse(
            OffsetDateTime.now(),
            status.value(),
            status.getReasonPhrase(),
            "An unexpected error occurred",
            request.getRequestURI()
        );
        logger.error(
            "api_exception type=GenericException status={} message={} path={}",
            status.value(),
            ex.getMessage(),
            request.getRequestURI(),
            ex
        );
        return ResponseEntity.status(status).body(errorResponse);
    }

}
