package com.alphaone.domain.service;

import com.alphaone.domain.repository.TopicRepository;
import com.alphaone.api.dto.TopicDto;

import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Set;

@Service
public class TopicService {
    private final TopicRepository topicRepository;

    private static final Set<String> GENERAL_TOPICS = Set.of("MACRO", "TECHNOLOGY");

    public TopicService(TopicRepository topicRepository) {
        this.topicRepository = topicRepository;
    }

    public List<TopicDto> getTrackedTopics() {
        return topicRepository.findAllByOrderBySlugAsc()
            .stream()
            .filter(topic -> !GENERAL_TOPICS.contains(topic.getSlug()))
            .map(topic -> new TopicDto(topic.getId(), topic.getSlug()))
            .toList();
    }
}
