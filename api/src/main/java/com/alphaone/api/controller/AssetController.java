package com.alphaone.api.controller;

import com.alphaone.domain.service.TopicService;
import com.alphaone.api.dto.TopicDto;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/assets")
public class AssetController {
    
    private final TopicService topicService;

    public AssetController(TopicService topicService) {
        this.topicService = topicService;
    }

    @GetMapping("/tracked")
    public List<TopicDto> getTrackedAssets() {
        return topicService.getTrackedTopics();
    }

}
