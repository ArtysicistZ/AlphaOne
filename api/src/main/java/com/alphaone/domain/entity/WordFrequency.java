package com.alphaone.domain.entity;

import java.time.LocalDate;

import jakarta.persistence.*;

@Entity
@Table(name = "word_frequency")
public class WordFrequency {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "word")
    private String word;

    @Column(name = "frequency")
    private Integer frequency;

    @Column(name = "date")
    private LocalDate date;

    // Getters and setters

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getWord() { return word; }
    public void setWord(String word) { this.word = word; }

    public Integer getFrequency() { return frequency; }
    public void setFrequency(Integer frequency) { this.frequency = frequency; }

    public LocalDate getDate() { return date; }
    public void setDate(LocalDate date) { this.date = date; }
    
}
