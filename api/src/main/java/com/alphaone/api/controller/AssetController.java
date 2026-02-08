package com.alphaone.api.controller;

import com.alphaone.domain.service.TopicService;
import com.alphaone.api.dto.TopicDto;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

@RestController
@RequestMapping("/api/v1/assets")
public class AssetController {
    
    private final TopicService topicService;
    private final Logger logger = LoggerFactory.getLogger(AssetController.class);

    public AssetController(TopicService topicService) {
        this.topicService = topicService;
    }

    @GetMapping("/tracked")
    public List<TopicDto> getTrackedAssets() {
        logger.info("api.assets.tracked.request");
        List<TopicDto> trackedTopics = topicService.getTrackedTopics();
        logger.info("api.assets.tracked.response count: {}", trackedTopics.size());
        
        return trackedTopics;
    }

}
