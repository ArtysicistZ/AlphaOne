package com.alphaone.domain.entity;

import java.util.Set;

import jakarta.persistence.*;
import jakarta.persistence.ManyToMany;

@Entity
@Table(name = "topics")
public class Topic {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "slug", unique = true)
    private String slug;

    @Column(name = "name")
    private String name;


    @ManyToMany(mappedBy = "topics")
    private Set<SentimentData> sentiments;

    // Getters and setters

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getSlug() { return slug; }
    public void setSlug(String slug) { this.slug = slug; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

}
