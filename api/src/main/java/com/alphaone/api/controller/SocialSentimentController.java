package com.alphaone.api.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;

import java.util.List;

import com.alphaone.domain.service.SentimentDataService;
import com.alphaone.domain.service.WordFrequencyService;
import com.alphaone.api.dto.DailySentimentDto;
import com.alphaone.api.dto.SentimentDataDto;
import com.alphaone.api.dto.WordCloudItemDto;


@RestController
@RequestMapping("/api/v1/signals/social-sentiment")
public class SocialSentimentController {
    
    private final SentimentDataService sentimentDataService;
    private final WordFrequencyService wordFrequencyService;

    public SocialSentimentController(
        SentimentDataService sentimentDataService,
        WordFrequencyService wordFrequencyService
    ) {
        this.sentimentDataService = sentimentDataService;
        this.wordFrequencyService = wordFrequencyService;
    }

    @GetMapping("/{ticker}/evidence")
    public List<SentimentDataDto> getEvidence(
        @PathVariable String ticker
    ) {
        return sentimentDataService.getEvidenceByTicker(ticker);
    }

    @GetMapping("/summary/{topicSlug}")
    public DailySentimentDto getTopicSummary(
        @PathVariable String topicSlug
    ) {
        return sentimentDataService.getTopicSummary(topicSlug);
    }

    @GetMapping("/wordcloud")
    public List<WordCloudItemDto> getWordCloud() {
        return wordFrequencyService.getWordCloudItems();
    }

    @GetMapping("/{ticker}/daily")
    public List<DailySentimentDto> getDailySentimentByTicker(
        @PathVariable String ticker
    ) {
        return sentimentDataService.getDailySentimentByTicker(ticker);
    }
    
}
