package com.alphaone.domain.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import com.alphaone.domain.entity.Topic;
import java.util.Optional;
import java.util.List;

public interface TopicRepository extends JpaRepository<Topic, Long> {
    Optional<Topic> findBySlug(String slug);
    List<Topic> findAllByOrderBySlugAsc();
}
