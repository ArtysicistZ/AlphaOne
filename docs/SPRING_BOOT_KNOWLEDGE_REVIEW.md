# Spring Boot Knowledge Review for alphaone

## 1) Purpose of this document
1. This document records core Spring Boot knowledge learned while migrating the API layer from FastAPI to Spring Boot.
2. It is designed for review before interviews and implementation sessions.
3. It is intentionally practical and tied to this codebase.
4. It focuses on backend API integration with PostgreSQL, not full Java ecosystem theory.
5. It includes architecture decisions, configuration details, and debugging patterns.

## 2) Project context in one page
1. The repo has three major parts: `backend/` (Python data pipeline), `api/` (Spring Boot API), and `frontend/` (React).
2. Python side still ingests Reddit data, processes sentiment, and writes to PostgreSQL.
3. Spring side now reads PostgreSQL and serves frontend endpoints.
4. Frontend can switch from FastAPI to Spring by changing `VITE_API_URL`.
5. The migration scope was API parity first, optimization later.

## 3) Big picture architecture
1. Ingestion and NLP are currently Python jobs.
2. Persistence is PostgreSQL.
3. Query API is Spring Boot.
4. UI is React/Vite.
5. Data flow is: raw social text -> Python processing -> DB rows -> Spring controllers -> JSON -> React components.
6. This separation improves system design clarity and resume signal.
7. It also reduces migration risk because only one layer changes at a time.

## 4) Why Spring Boot for this API
1. Spring Boot provides strong structure for layered services.
2. Spring Data JPA reduces boilerplate for CRUD and query methods.
3. Spring MVC annotations make route mapping explicit and readable.
4. Spring ecosystem has production-grade patterns for observability and security.
5. Recruiters often recognize Spring Boot architecture experience.

## 5) What Spring Initializr is
1. Spring Initializr is a project scaffold generator.
2. It creates a build file (`pom.xml` or Gradle), app entrypoint, and folder structure.
3. It lets you select dependencies up front.
4. It reduces setup mistakes compared with manual bootstrap.
5. It is commonly used by teams to start new Spring services quickly.

## 6) Packaging choice
1. `JAR` is standard for modern Spring microservices.
2. `WAR` is older model for external app servers.
3. This project uses `JAR`.
4. `JAR` works well for local run, Docker, and cloud deployment.

## 7) Where Spring service lives in this repo
1. Spring project is in top-level `api/`.
2. Keeping it outside `backend/` avoids mixing Java and Python concerns.
3. Clean layout improves maintainability and onboarding.
4. Suggested root layout is: `backend/`, `api/`, `frontend/`, `docs/`.

## 8) Core dependencies used
1. `spring-boot-starter-web` for REST controllers.
2. `spring-boot-starter-data-jpa` for ORM and repositories.
3. `postgresql` driver for DB connectivity.
4. `spring-boot-starter-actuator` for health and metrics endpoints.
5. `spring-boot-starter-validation` for request validation support.
6. `spring-dotenv` to load `.env` into Spring property sources.

## 9) Spring configuration basics
1. Main config file is `api/src/main/resources/application.yml`.
2. Data source points to `${DATABASE_URL}` and related vars.
3. `spring.jpa.hibernate.ddl-auto=none` prevents schema modifications.
4. `spring.jpa.show-sql=true` helps during debugging.
5. Production configs should use profiles and safer logging settings.

## 10) `.env` loading lessons
1. Spring Boot does not natively read `.env` files.
2. `spring-dotenv` was added to support `.env` loading.
3. `.env` needs to be in the Spring working directory (for this setup, `api/.env`).
4. Python and Spring can keep separate `.env` files.
5. Never commit real secrets; rotate exposed credentials immediately.

## 11) JDBC URL lessons
1. Spring/Hikari expects JDBC URL format: `jdbc:postgresql://...`.
2. Python URL style `postgresql://...` is not accepted by Hikari.
3. Error `"url must start with jdbc"` indicates wrong URL prefix.
4. Error `"Driver ... does not accept jdbcUrl"` can happen if URL/user parsing is inconsistent.
5. Splitting username/password into separate properties can stabilize parsing.

## 12) Database credentials mapping
1. `DATABASE_URL` should use JDBC format.
2. `DATABASE_USERNAME` and `DATABASE_PASSWORD` can be separate.
3. For Neon and similar providers, SSL parameters may be required in URL.
4. Keep these in `.env`, never hardcode in Java files.

## 13) App entrypoint and scanning
1. Spring app entry class is `ApiApplication`.
2. `@SpringBootApplication` enables auto-config and scanning.
3. Custom base scanning can be set via `scanBasePackages`.
4. Explicit JPA scanning can be added with:
5. `@EnableJpaRepositories(basePackages = "...")`
6. `@EntityScan(basePackages = "...")`
7. This avoids bean discovery issues when package layout is split.

## 14) Layered architecture in this codebase
1. `entity` layer maps DB tables.
2. `repository` layer defines DB query contracts.
3. `service` layer implements business logic.
4. `controller` layer handles HTTP request/response mapping.
5. `dto` layer defines API payload shapes.
6. `config` layer contains global app configs (like CORS).

## 15) Why this layered structure matters
1. It isolates concerns.
2. It prevents controllers from containing SQL or heavy logic.
3. It improves testability by mocking repositories in service tests.
4. It supports future refactors with lower risk.
5. It is standard in enterprise Java teams.

## 16) Entities created for this migration
1. `Topic` for `topics` table.
2. `SentimentData` for `sentiment_data` table.
3. `WordFrequency` for `word_frequency` table.
4. Relationship mapping for join table `sentiment_topic_association`.

## 17) JPA entity annotation essentials
1. `@Entity` marks class as JPA-managed entity.
2. `@Table(name = "...")` binds class to DB table.
3. `@Id` marks primary key.
4. `@GeneratedValue(strategy = GenerationType.IDENTITY)` maps identity generation.
5. `@Column(name = "...")` maps Java fields to DB columns.

## 18) Many-to-many mapping explained
1. One sentiment record can mention multiple topics.
2. One topic appears across many sentiment records.
3. Therefore relationship is many-to-many.
4. JPA mapping on `SentimentData` uses `@ManyToMany` and `@JoinTable`.
5. `Topic` side uses `@ManyToMany(mappedBy = "topics")`.
6. Join table columns are `sentiment_data_id` and `topic_id`.

## 19) Why getters/setters were needed
1. Services map entity fields into DTOs.
2. Accessing private fields requires methods like `getId()`.
3. Compile errors like `getId() undefined` indicate missing accessors.
4. Later, Lombok can reduce boilerplate if desired.

## 20) Repository concept
1. A repository is a data-access contract.
2. It defines what queries can be performed.
3. It does not include HTTP logic.
4. It should not include DTO mapping.
5. Spring creates runtime implementations for repository interfaces.

## 21) `JpaRepository` benefits
1. Built-in CRUD methods (`findAll`, `findById`, `save`, `delete`).
2. Pagination and sorting support.
3. No manual implementation for common operations.
4. Easy extension with custom query methods.

## 22) Derived query method rules
1. Method names are parsed by Spring Data.
2. `findBySlug` means `WHERE slug = ?`.
3. `findAllByOrderBySlugAsc` means sorted full query.
4. Names must reference entity field names, not necessarily DB column names.
5. Types must match field types.

## 23) Repository methods used in this migration
1. `TopicRepository.findBySlug(...)`
2. `TopicRepository.findAllByOrderBySlugAsc()`
3. `SentimentDataRepository.findTop5ByTopicsContainsOrderByCreatedAtDesc(...)`
4. `SentimentDataRepository.findByTopicsContains(...)`
5. `WordFrequencyRepository.findTop100ByOrderByFrequencyDesc()`

## 24) Service layer concept
1. Services implement use-case logic.
2. Services orchestrate repositories and mapping.
3. Controllers call services.
4. Services should be small and focused by domain.
5. This migration used `TopicService`, `SentimentDataService`, and `WordFrequencyService`.

## 25) DTO concept
1. DTO = Data Transfer Object.
2. DTOs represent API contracts, not DB models.
3. Returning entities directly risks recursive JSON and data leakage.
4. DTOs keep responses stable when internal schema changes.
5. DTOs are ideal for frontend-specific payload shapes.

## 26) Java `record` concept
1. `record` is a compact immutable data carrier.
2. Java auto-generates constructor/accessors/equals/hashCode/toString.
3. Accessors are field-style (`id()`) not `getId()`.
4. Good fit for read-only response DTOs.
5. Keep constructor parameter types aligned with caller code.

## 27) Date/time type lessons
1. `LocalDate` is date only (no time, no timezone).
2. `OffsetDateTime` includes date, time, and UTC offset.
3. Use `LocalDate` for daily aggregate outputs.
4. Use `OffsetDateTime` for row-level timestamps.
5. Wrong imports (for example Spring internal `Local`) break constructors.

## 28) Controller mapping fundamentals
1. `@RestController` marks HTTP controller.
2. `@RequestMapping` defines base route.
3. `@GetMapping` defines GET handler path.
4. `@PathVariable` binds URL path segment to method parameter.
5. Spring builds route table at startup from these annotations.

## 29) Endpoints implemented in Spring
1. `GET /api/v1/assets/tracked`
2. `GET /api/v1/signals/social-sentiment/{ticker}/evidence`
3. `GET /api/v1/signals/social-sentiment/summary/{topicSlug}`
4. `GET /api/v1/signals/social-sentiment/{ticker}/daily`
5. `GET /api/v1/signals/social-sentiment/wordcloud`

## 30) Route parity strategy
1. First target was endpoint parity with FastAPI.
2. This reduces frontend changes during migration.
3. Once parity is stable, optimize internals.
4. This sequencing minimizes migration risk.

## 31) Error handling approach used
1. Service throws `ResponseStatusException` for not found topics.
2. This maps naturally to HTTP 404 responses.
3. Next step can centralize error format with `@ControllerAdvice`.
4. Consistent error JSON helps frontend robustness.

## 32) CORS concept
1. Browser enforces same-origin policy.
2. Frontend (`:5173`) and API (`:8080`) are different origins.
3. Without CORS headers, browser blocks responses.
4. CORS config enables allowed origins/methods/headers for `/api/**`.
5. `allowCredentials(true)` requires explicit origins, not `*`.

## 33) CORS implementation details
1. Added `WebMvcConfigurer` bean in `CorsConfig`.
2. Implemented `addCorsMappings(...)` for API routes.
3. Allowed local dev and production frontend origins.
4. Included allowed methods and headers.
5. Added `@NonNull` annotation for signature compatibility warning.

## 34) Stream API concept
1. `.stream()` creates a processing pipeline over collections.
2. `.map(...)` transforms each element.
3. `.filter(...)` removes unwanted elements.
4. `.toList()` collects back to list.
5. This style replaced manual loops in DTO mapping and aggregation.

## 35) `collect` and `Collectors` concept
1. `.collect(...)` is a terminal stream operation.
2. `Collectors` provides common aggregation recipes.
3. `groupingBy(...)` groups rows by key.
4. `summarizingDouble(...)` computes count/sum/min/max/average.
5. Equivalent conceptually to pandas `groupby().agg(...)`.

## 36) Daily aggregation behavior
1. Current daily endpoint groups sentiment rows by `createdAt.toLocalDate()`.
2. It computes average score per day.
3. Output is sorted by day using `TreeMap`.
4. This is an in-memory aggregation approach.
5. Later optimization can move aggregation into SQL.

## 37) Summary endpoint behavior
1. Current summary endpoint computes one average over all topic rows.
2. It returns `day = LocalDate.now()` and `averageScore`.
3. This matches original FastAPI behavior during parity stage.
4. Future refinement can make this strictly “today only” if required.

## 38) Word cloud endpoint behavior
1. Current logic returns top 100 by frequency descending.
2. Output DTO shape is `{ text, value }`.
3. Data source is `word_frequency` table.
4. No date filter currently if repository method is global top 100.
5. Future change can add date filter for “today’s words only.”

## 39) Naming consistency lesson
1. A DTO naming typo (`WorldCloudItemDto`) can compile but create confusion.
2. Naming consistency reduces maintenance errors.
3. Rename carefully and update imports in all callers.
4. Do this after functionality is verified.

## 40) Testing checklist for each new endpoint
1. Start API service and ensure app boots.
2. Check `/actuator/health` is `UP`.
3. Hit endpoint with known valid topic/ticker.
4. Hit endpoint with invalid topic/ticker and verify expected error status.
5. Validate field names and types match frontend expectations.

## 41) Common debugging pattern used
1. Confirm route path is exact (including trailing slash behavior).
2. Confirm controller method is being hit.
3. Confirm service input lookup (topic slug) returns entity.
4. Confirm repository method returns rows.
5. Confirm DTO mapping fields are non-null and type-safe.

## 42) High-value errors and what they mean
1. `url must start with "jdbc"` means non-JDBC DB URL.
2. `No qualifying bean ... Repository` means package scan/repository scan mismatch.
3. `constructor ... undefined` in DTO means type/signature mismatch.
4. CORS browser error with backend 200 means missing CORS headers.
5. Whitelabel 404 can mean wrong path or handler not mapped.

## 43) Frontend integration switch
1. Frontend axios base URL reads from `VITE_API_URL`.
2. Use `frontend/.env.local` with `VITE_API_URL=http://127.0.0.1:8080`.
3. Restart Vite after env change.
4. Run smoke test on key UI pages.
5. Keep FastAPI available as temporary fallback if needed.

## 44) Migration strategy lessons
1. Migrate one vertical slice first (assets endpoint).
2. Verify end-to-end before adding complexity.
3. Then add evidence, summary, wordcloud, daily endpoints.
4. Delay performance optimization until correctness is stable.
5. Keep old service running during migration to reduce risk.

## 45) Why this approach is good for resume
1. Demonstrates incremental migration strategy.
2. Demonstrates service layering and API contract control.
3. Shows practical debugging and reliability thinking.
4. Shows framework depth beyond copy-paste setup.
5. Produces measurable deliverables and architecture clarity.

## 46) Security and secret hygiene reminders
1. Never commit real DB/API credentials.
2. Rotate any exposed credentials immediately.
3. Keep `.env` out of git history.
4. Prefer environment injection in deployment platform.
5. Add `.env.example` templates for onboarding.

## 47) Performance considerations for next iteration
1. In-memory group-by is simple but can be expensive with large history.
2. DB-side aggregation with SQL can reduce memory and response time.
3. Add indexes for frequent query dimensions.
4. Cache high-traffic endpoints.
5. Measure before and after optimization.

## 48) Data model caveats to remember
1. `many-to-many` joins can become expensive at scale.
2. Row-level evidence endpoint should cap result count.
3. Summary semantics must be explicit (all-time vs today).
4. Word cloud semantics must be explicit (global vs date-scoped).
5. Avoid ambiguous API names that hide time-window behavior.

## 49) Recommended near-term improvements
1. Add `@ControllerAdvice` with consistent error payload.
2. Add integration tests for all 5 endpoints.
3. Normalize DTO names and package naming.
4. Add request/response logging with correlation IDs.
5. Add OpenAPI docs for contract visibility.

## 50) Recommended medium-term system design upgrades
1. Introduce message broker between ingestion and processing.
2. Convert heavy aggregations to precomputed tables.
3. Add distributed cache for read-heavy endpoints.
4. Add metrics dashboard and alerting.
5. Add containerized multi-service local stack.

## 51) Practical interview talking points from this migration
1. Why parity-first migration reduces blast radius.
2. Why DTO separation matters for API stability.
3. How you debugged JDBC and bean wiring failures.
4. How you mapped many-to-many sentiment-topic relationships.
5. How CORS impacts browser-facing system integration.

## 52) Review Q&A quick drill
1. Q: Why not return JPA entities directly?
2. A: Risk of recursion, lazy-loading issues, and leaking internal fields.
3. Q: What does `findBySlug` do?
4. A: Derived query mapping to `WHERE slug = ?`.
5. Q: Why use `LocalDate` in daily DTO?
6. A: Daily aggregate needs date-only semantics.
7. Q: Why did Spring fail with PostgreSQL URL initially?
8. A: URL was not JDBC format or had parse issues.
9. Q: Why add explicit repo/entity scanning?
10. A: Package layout prevented automatic discovery in default scan path.
11. Q: What does CORS solve?
12. A: Browser cross-origin request permission.
13. Q: Why use service layer?
14. A: To keep business logic out of controllers and ease testing.
15. Q: Why use records for DTOs?
16. A: Immutable, concise, generated constructor/accessors.
17. Q: What does `collect(groupingBy(...))` do?
18. A: Group stream items and aggregate by key.
19. Q: Summary vs daily endpoint difference?
20. A: Summary returns one aggregate; daily returns time series.

## 53) Interview-ready architecture statement
1. “I migrated the API layer from FastAPI to Spring Boot while preserving endpoint contracts.”
2. “I modeled existing PostgreSQL schema with JPA entities, including a many-to-many join.”
3. “I used repository-service-controller layering with DTO boundaries for clean contracts.”
4. “I solved real integration issues around JDBC URL parsing, package scanning, and CORS.”
5. “I validated endpoint parity and staged the frontend switch via environment config.”

## 54) Detailed endpoint contract notes
1. Assets endpoint returns a list of tracked topic slugs for UI choices.
2. Evidence endpoint returns last five topic-linked sentiment rows.
3. Summary endpoint currently computes all-time average by topic.
4. Daily endpoint returns average score per calendar day.
5. Wordcloud endpoint returns top-weighted tokens as `text/value`.

## 55) Reliability notes
1. App startup health should be checked each run.
2. DB connectivity should fail fast and clearly.
3. Repository errors should map to meaningful HTTP responses.
4. CORS should be explicit and environment-aware.
5. Logs should capture route hits and failures with context.

## 56) Data correctness notes
1. Topic lookup currently uppercases ticker in some flows.
2. Case handling policy should be consistent across endpoints.
3. Empty result vs not-found must be intentionally differentiated.
4. Null sentiment scores should be filtered before averaging.
5. Date extraction should align with timezone policy.

## 57) Code quality notes
1. Keep class names and DTO names consistent.
2. Avoid duplicate DTO definitions.
3. Keep methods small and single-purpose.
4. Use explicit imports for clarity.
5. Add concise comments only where logic is non-obvious.

## 58) Suggested test cases (backend)
1. Controller returns 200 for known ticker.
2. Controller returns 404 for unknown ticker.
3. Repository derived queries return expected counts.
4. Service average calculation handles empty list correctly.
5. CORS headers exist for allowed origin requests.

## 59) Suggested test cases (integration)
1. DB seed with known rows and known averages.
2. Verify `/daily` numeric outputs are accurate.
3. Verify `/summary` matches expected average.
4. Verify `/wordcloud` order and max size.
5. Verify frontend can render all endpoint responses.

## 60) Final review checklist before presenting this project
1. Endpoints all pass manual smoke tests.
2. Frontend points to Spring and works in browser.
3. `.env` secrets are not committed.
4. Documentation reflects current architecture and status.
5. At least one architecture diagram exists in docs.
6. Known limitations are documented honestly.
7. Next steps are prioritized by ROI and effort.
8. You can explain each Spring annotation used.
9. You can explain each repository method and return type.
10. You can explain each endpoint’s time window semantics.

## 61) Minimal glossary
1. Spring Boot: opinionated framework for Java services.
2. JPA: Java Persistence API, ORM abstraction.
3. Hibernate: common JPA implementation.
4. Repository: data-access contract layer.
5. Service: business logic layer.
6. Controller: HTTP interface layer.
7. DTO: API payload object.
8. CORS: browser cross-origin access control.
9. JDBC: Java database connectivity protocol.
10. Actuator: operational endpoints for Spring apps.

## 62) Short retrospective from this migration
1. The most common blockers were configuration mismatches, not business logic.
2. Layer-by-layer implementation prevented large debugging surfaces.
3. Endpoint parity first was the correct strategy.
4. Strong naming and type consistency matters in Java.
5. Documentation during migration reduced confusion and rework.

## 63) One-page implementation summary
1. Spring app scaffolded in `api/`.
2. DB connectivity established using `.env` and JDBC URL.
3. JPA entities mapped to existing Python-created schema.
4. Repositories implemented with derived query methods.
5. Services mapped entities to DTO records.
6. Controllers exposed API parity routes.
7. CORS configured for browser integration.
8. Endpoints manually validated.
9. Plan updated for next phases (testing, observability, optimization).

## 64) Keep-learning path (practical)
1. Learn `@Query` and projections for SQL-side aggregation.
2. Learn pagination and sorting via `Pageable`.
3. Learn transaction boundaries with `@Transactional`.
4. Learn global exception handling with `@ControllerAdvice`.
5. Learn testing with `@WebMvcTest` and `@DataJpaTest`.
6. Learn observability with Micrometer and Prometheus.
7. Learn containerization and environment profiles.

## 65) Final note
1. This document is a living review guide.
2. Update it each time architecture or endpoint semantics change.
3. Keep it aligned with `docs/FUTURE_PLAN.md`.
4. Prefer concrete examples over abstract statements.
5. If a behavior changes, record both old and new behavior explicitly.
