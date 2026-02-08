package com.alphaone.domain.service;

import com.alphaone.domain.repository.SentimentDataRepository;
import com.alphaone.domain.repository.TopicRepository;
import com.alphaone.domain.entity.SentimentData;
import com.alphaone.domain.entity.Topic;
import com.alphaone.api.dto.DailySentimentDto;
import com.alphaone.api.dto.SentimentDataDto;

import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;
import java.util.List;
import java.util.DoubleSummaryStatistics;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

@Service
public class SentimentDataService {

    private final TopicRepository topicRepository;
    private final SentimentDataRepository sentimentDataRepository;


    public SentimentDataService(
        TopicRepository topicRepository, SentimentDataRepository sentimentDataRepository
    ) {
        this.topicRepository = topicRepository;
        this.sentimentDataRepository = sentimentDataRepository;
    }


    public List<SentimentDataDto> getEvidenceByTicker(String ticker) {
        Topic topic = 
            topicRepository
                .findBySlug(ticker.toUpperCase())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Topic not found for ticker: " + ticker));

        return sentimentDataRepository.findTop5ByTopicsContainsOrderByCreatedAtDesc(topic)
            .stream()
            .map(data -> new SentimentDataDto(
                data.getId(),
                data.getCreatedAt(),
                data.getSentimentLabel(),
                data.getSentimentScore(),
                data.getRelevantText()
            ))
            .toList();
    }


    public DailySentimentDto getTopicSummary(String topicSlug) {
        Topic topic = 
            topicRepository
                .findBySlug(topicSlug.toUpperCase())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Topic not found for slug: " + topicSlug));
        
        List<SentimentData> dataList = sentimentDataRepository.findByTopicsContains(topic);

        double avg = dataList.stream()
            .map(SentimentData::getSentimentScore)
            .filter(java.util.Objects::nonNull)
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);

        return new DailySentimentDto(java.time.LocalDate.now(), avg);
    }


    public List<DailySentimentDto> getDailySentimentByTicker(String ticker) {
        Topic topic = 
            topicRepository
                .findBySlug(ticker.toUpperCase())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Topic not found for ticker: " + ticker));
        
        List<SentimentData> dataList = sentimentDataRepository.findByTopicsContains(topic);

        Map<java.time.LocalDate, DoubleSummaryStatistics> grouped = 
            dataList.stream()
                .filter(
                    data -> data.getCreatedAt() != null && 
                    data.getSentimentScore() != null
                )
                .collect(
                    Collectors.groupingBy(
                        data -> data.getCreatedAt().toLocalDate(),
                        TreeMap::new,
                        Collectors.summarizingDouble(SentimentData::getSentimentScore)
                    )
                );

        return grouped
            .entrySet()
            .stream()
            .map(
                entry -> new DailySentimentDto(entry.getKey(), 
                entry.getValue().getAverage())
            )
            .toList();
    }
    
}
