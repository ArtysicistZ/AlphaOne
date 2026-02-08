package com.alphaone.api;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;

@SpringBootTest
@AutoConfigureMockMvc
class ApiTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void trackedAssets_shouldReturn200() throws Exception {
        mockMvc.perform(get("/api/v1/assets/tracked"))
                .andExpect(status().isOk());
    }

    @Test
    void wordCloud_shouldReturn200AndJson() throws Exception {
        mockMvc.perform(get("/api/v1/signals/social-sentiment/wordcloud"))
                .andExpect(status().isOk())
                .andExpect(content().contentTypeCompatibleWith("application/json"));
    }

    @Test
    void unknownSummary_shouldReturn404() throws Exception {
        mockMvc.perform(get("/api/v1/signals/social-sentiment/summary/UNKNOWN_TOPIC_XYZ"))
                .andExpect(status().isNotFound());
    }
}
