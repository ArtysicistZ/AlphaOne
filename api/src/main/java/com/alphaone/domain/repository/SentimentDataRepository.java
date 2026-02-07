package com.alphaone.domain.repository;

import com.alphaone.domain.entity.SentimentData;
import com.alphaone.domain.entity.Topic;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

public interface SentimentDataRepository extends JpaRepository<SentimentData, Long> {
    List<SentimentData> findTop5ByTopicsContainsOrderByCreatedAtDesc(Topic topic);
    List<SentimentData> findByTopicsContains(Topic topic);
}
