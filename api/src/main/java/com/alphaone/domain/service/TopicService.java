package com.alphaone.domain.service;

import com.alphaone.domain.repository.TopicRepository;
import com.alphaone.api.dto.TopicDto;

import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class TopicService {
    private final TopicRepository topicRepository;

    public TopicService(TopicRepository topicRepository) {
        this.topicRepository = topicRepository;
    }

    public List<TopicDto> getTrackedTopics() {
        return topicRepository.findAllByOrderBySlugAsc()
            .stream()
            .map(topic -> new TopicDto(topic.getId(), topic.getSlug()))
            .toList();
    }
}
