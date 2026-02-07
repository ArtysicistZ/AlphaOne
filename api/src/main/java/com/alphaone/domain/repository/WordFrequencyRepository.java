package com.alphaone.domain.repository;

import com.alphaone.domain.entity.WordFrequency;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface WordFrequencyRepository extends JpaRepository<WordFrequency, Long> {
    List<WordFrequency> findTop100ByOrderByFrequencyDesc();
}
