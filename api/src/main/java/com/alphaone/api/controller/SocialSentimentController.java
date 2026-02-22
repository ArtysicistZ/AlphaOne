package com.alphaone.api.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import com.alphaone.domain.service.SentimentDataService;
import com.alphaone.api.dto.DailySentimentDto;
import com.alphaone.api.dto.SentimentDataDto;


@RestController
@RequestMapping("/api/v1/signals/social-sentiment")
public class SocialSentimentController {
    
    private final SentimentDataService sentimentDataService;

    private static final Logger logger = LoggerFactory.getLogger(SocialSentimentController.class);

    public SocialSentimentController(SentimentDataService sentimentDataService) {
        this.sentimentDataService = sentimentDataService;
    }

    @GetMapping("/{ticker}/evidence")
    public List<SentimentDataDto> getEvidence(
        @PathVariable String ticker
    ) {
        logger.info("api.social.evidence.request ticker={}", ticker);
        List<SentimentDataDto> result = sentimentDataService.getEvidenceByTicker(ticker);
        logger.info("api.social.evidence.response ticker={} count={}", ticker, result.size());
        return result;
    }

    @GetMapping("/macro/summary")
    public DailySentimentDto getMacroSummary() {
        logger.info("api.social.macro.summary.request");
        DailySentimentDto result = sentimentDataService.getMacroSummary();
        logger.info("api.social.macro.summary.response avg={}", result.averageScore());
        return result;
    }

    @GetMapping("/macro/daily")
    public List<DailySentimentDto> getMacroDailyChart() {
        logger.info("api.social.macro.daily.request");
        List<DailySentimentDto> result = sentimentDataService.getMacroDailyChart();
        logger.info("api.social.macro.daily.response count={}", result.size());
        return result;
    }

    @GetMapping("/summary/{topicSlug}")
    public DailySentimentDto getTopicSummary(
        @PathVariable String topicSlug
    ) {
        logger.info("api.social.summary.request topicSlug={}", topicSlug);
        DailySentimentDto result = sentimentDataService.getTopicSummary(topicSlug);
        logger.info("api.social.summary.response topicSlug={}", topicSlug);
        return result;
    }

    @GetMapping("/{ticker}/daily")
    public List<DailySentimentDto> getDailySentimentByTicker(
        @PathVariable String ticker
    ) {
        logger.info("api.social.daily.request ticker={}", ticker);
        List<DailySentimentDto> result = sentimentDataService.getDailySentimentByTicker(ticker);
        logger.info("api.social.daily.response ticker={} count={}", ticker, result.size());
        return result;
    }
    
}
