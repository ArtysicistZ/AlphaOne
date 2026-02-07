package com.alphaone.domain.entity;

import java.util.Set;
import java.time.OffsetDateTime;

import jakarta.persistence.*;
import jakarta.persistence.ManyToMany;
import jakarta.persistence.JoinTable;
import jakarta.persistence.JoinColumn;


@Entity
@Table(name = "sentiment_data")
public class SentimentData {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "created_at")
    private OffsetDateTime createdAt;

    @Column(name = "source_id", unique = true)
    private String sourceId;

    @Column(name = "source_type")
    private String sourceType;

    @Column(name = "sentiment_label")
    private String sentimentLabel;

    @Column(name = "sentiment_score")
    private Double sentimentScore;

    @Column(name = "relevant_text")
    private String relevantText;

    @ManyToMany
    @JoinTable(
        name = "sentiment_topic_association",
        joinColumns = @JoinColumn(name = "sentiment_data_id"),
        inverseJoinColumns = @JoinColumn(name = "topic_id")
    )
    private Set<Topic> topics;

    // Getters and setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public OffsetDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(OffsetDateTime createdAt) { this.createdAt = createdAt; }

    public String getSourceId() { return sourceId; }
    public void setSourceId(String sourceId) { this.sourceId = sourceId; }

    public String getSourceType() { return sourceType; }
    public void setSourceType(String sourceType) { this.sourceType = sourceType; }

    public String getSentimentLabel() { return sentimentLabel; }
    public void setSentimentLabel(String sentimentLabel) { this.sentimentLabel = sentimentLabel; }

    public Double getSentimentScore() { return sentimentScore; }
    public void setSentimentScore(Double sentimentScore) { this.sentimentScore = sentimentScore; }

    public String getRelevantText() { return relevantText; }
    public void setRelevantText(String relevantText) { this.relevantText = relevantText; }

    public Set<Topic> getTopics() { return topics; }
    public void setTopics(Set<Topic> topics) { this.topics = topics; }

}
