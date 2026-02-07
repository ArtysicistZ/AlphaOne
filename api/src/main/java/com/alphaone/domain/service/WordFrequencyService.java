package com.alphaone.domain.service;

import com.alphaone.domain.repository.WordFrequencyRepository;
import com.alphaone.api.dto.WordCloudItemDto;

import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class WordFrequencyService {
    private final WordFrequencyRepository wordFrequencyRepository;

    public WordFrequencyService(WordFrequencyRepository wordFrequencyRepository) {
        this.wordFrequencyRepository = wordFrequencyRepository;
    }

    public List<WordCloudItemDto> getWordCloudItems() {
        return wordFrequencyRepository
            .findTop100ByOrderByFrequencyDesc()
            .stream()
            .map(wf -> new WordCloudItemDto(wf.getWord(), wf.getFrequency()))
            .toList();
    }

}
