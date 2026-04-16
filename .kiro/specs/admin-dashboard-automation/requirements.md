# Requirements Document

## Introduction

This feature covers the automated test suite for the Admin Dashboard of the Owlet Campus web application (`https://owlet-campus.com/admin`). The suite is built with Playwright and JavaScript, following the Page Object Model (POM) pattern. It converts the existing manual test flow — covering admin login, course creation, curriculum management, and lesson type configuration — into maintainable, production-ready automated scripts.

The suite is organised into three describe blocks: **Authentication**, **Course Management**, and **Curriculum Management**. Reusable helper functions (`login()`, `createCourse()`, `addModule()`, `addLesson()`) are extracted into shared utilities to keep individual test cases concise and independent.

## Glossary

- **Test_Suite**: The complete collection of Playwright test files targeting the Owlet Campus admin panel.
- **Admin_Login_Page**: The Page Object representing the admin login form at `/admin/login`, with Email and Password fields and a Login button.
- **Dashboard_Page**: The Page Object representing the authenticated admin dashboard, including the sidebar navigation and user identity display.
- **Courses_Page**: The Page Object representing the course listing at `/admin/courses`, including the course list and the "New Course" button.
- **Course_Form_Page**: The Page Object representing the new/edit course form, with fields for Title, Tagline, Description, Price, Duration, Category, Difficulty, and Thumbnail.
- **Curriculum_Page**: The Page Object representing the curriculum builder at `/admin/courses/{id}/edit`, including the curriculum map and module/lesson sidebar.
- **Lesson_Editor_Page**: The Page Object representing the lesson configuration editor for Video, Reading, Quiz, and Coding Lab lesson types.
- **POM**: Page Object Model — a design pattern that encapsulates page selectors and interactions into reusable class objects.
- **Module**: A top-level grouping of lessons within a course curriculum.
- **Lesson**: A single learning unit within a Module, typed as Video, Reading, Quiz, or Coding Lab.
- **Valid_Admin_Credentials**: The email/password pair for the admin account accepted by the admin login flow.
- **Dynamic_Course_Name**: A course title generated at runtime (e.g., with a timestamp suffix) to avoid collisions between test runs.
- **Locator**: A Playwright `page.locator()`, `getByRole()`, or `getByLabel()` reference used to identify a UI element.
- **Assertion**: A Playwright `expect()` call that verifies the state of the application.

---

## Requirements

### Requirement 1: Project Structure and Configuration

**User Story:** As a QA engineer, I want a well-structured Playwright project for admin automation, so that the test suite is maintainable, scalable, and easy to extend.

#### Acceptance Criteria

1. THE Test_Suite SHALL be organised using the Page Object Model pattern, with page objects stored in a `pages/` directory and test specs stored in a `tests/` directory.
2. THE Test_Suite SHALL use `async/await` syntax throughout all test files and page objects.
3. THE Test_Suite SHALL include a `playwright.config.js` file that sets the base URL to `https://owlet-campus.com/`.
4. THE Test_Suite SHALL use `getByRole()`, `getByLabel()`, `getByText()`, or `getByPlaceholder()` for element selection, preferring semantic locators over CSS selectors.
5. THE Test_Suite SHALL capture a screenshot on test failure using Playwright's `screenshot` option set to `'only-on-failure'`.
6. THE Test_Suite SHALL include inline comments explaining the purpose of each logical block.
7. THE Test_Suite SHALL group tests using `test.describe()` blocks named "Authentication", "Course Management", and "Curriculum Management".

---

### Requirement 2: Admin Login Page Object

**User Story:** As a QA engineer, I want a reusable Admin Login page object, so that authentication interactions are not duplicated across test cases.

#### Acceptance Criteria

1. THE Admin_Login_Page SHALL expose a `navigate()` method that navigates to `/admin/login`.
2. THE Admin_Login_Page SHALL expose a `fillForm(email, password)` method that fills the Email and Password fields.
3. THE Admin_Login_Page SHALL expose a `submit()` method that clicks the Login button.
4. THE Admin_Login_Page SHALL expose a `login(email, password)` convenience method that calls `fillForm` then `submit`.
5. THE Admin_Login_Page SHALL expose locator accessors for the Email field, Password field, and Login button.
6. IF a selector cannot be resolved at runtime, THEN THE Admin_Login_Page SHALL surface the Playwright locator error without swallowing it.

---

### Requirement 3: Dashboard Page Object

**User Story:** As a QA engineer, I want a Dashboard page object, so that post-login state assertions are reusable across test cases.

#### Acceptance Criteria

1. THE Dashboard_Page SHALL expose a `waitForLoad()` method that waits until the sidebar navigation is visible.
2. THE Dashboard_Page SHALL expose a `navigateTo(sectionName)` method that clicks the matching sidebar link by its visible text.
3. THE Dashboard_Page SHALL expose a locator accessor for the sidebar navigation element.
4. THE Dashboard_Page SHALL expose a locator accessor for the user identity display in the top-right area of the page.
5. IF the sidebar is not visible within the configured action timeout, THEN THE Dashboard_Page SHALL surface the Playwright timeout error without swallowing it.

---

### Requirement 4: Courses Page Object

**User Story:** As a QA engineer, I want a Courses page object, so that course listing interactions are encapsulated and reusable.

#### Acceptance Criteria

1. THE Courses_Page SHALL expose a `navigate()` method that navigates to `/admin/courses`.
2. THE Courses_Page SHALL expose a `clickNewCourse()` method that clicks the "New Course" button.
3. THE Courses_Page SHALL expose a `searchCourse(title)` method that locates a course card by its title text.
4. THE Courses_Page SHALL expose a `clickEditCourse(title)` method that clicks the edit (pen) icon on the course card matching the given title.
5. THE Courses_Page SHALL expose locator accessors for the course list container and the "New Course" button.

---

### Requirement 5: Course Form Page Object

**User Story:** As a QA engineer, I want a Course Form page object, so that course creation and editing interactions are encapsulated.

#### Acceptance Criteria

1. THE Course_Form_Page SHALL expose a `fillCourseDetails(details)` method that fills Title, Tagline, Description, Price, Duration, Category, and Difficulty fields from a details object.
2. THE Course_Form_Page SHALL expose an `uploadThumbnail(filePath)` method that uploads an image file to the thumbnail input.
3. THE Course_Form_Page SHALL expose a `saveAndBuildCurriculum()` method that clicks the "Save and Build Curriculum" button.
4. THE Course_Form_Page SHALL expose locator accessors for each form field: Title, Tagline, Description, Price, Duration, Category, Difficulty, and Thumbnail.
5. IF the "Save and Build Curriculum" button is not visible, THEN THE Course_Form_Page SHALL surface the Playwright locator error without swallowing it.

---

### Requirement 6: Curriculum Page Object

**User Story:** As a QA engineer, I want a Curriculum page object, so that module and lesson management interactions are encapsulated.

#### Acceptance Criteria

1. THE Curriculum_Page SHALL expose an `addModule(moduleName)` method that clicks "Add Module", enters the module name, and confirms the action.
2. THE Curriculum_Page SHALL expose an `addLesson(moduleName, lessonName, lessonType)` method that clicks "Add Lesson" under the specified module, fills the lesson name, selects the lesson type, and clicks "Create & Configure".
3. THE Curriculum_Page SHALL expose a `getModuleLocator(moduleName)` method that returns a Locator for the module entry in the curriculum sidebar.
4. THE Curriculum_Page SHALL expose a `getLessonLocator(lessonName)` method that returns a Locator for the lesson entry in the curriculum sidebar.
5. THE Curriculum_Page SHALL expose a locator accessor for the curriculum map container.
6. IF the curriculum map is not visible after navigation, THEN THE Curriculum_Page SHALL surface the Playwright timeout error without swallowing it.

---

### Requirement 7: Lesson Editor Page Object

**User Story:** As a QA engineer, I want a Lesson Editor page object, so that lesson content configuration interactions are encapsulated for all four lesson types.

#### Acceptance Criteria

1. THE Lesson_Editor_Page SHALL expose a `configureVideoLesson(videoUrl)` method that enters the video URL and saves the lesson.
2. THE Lesson_Editor_Page SHALL expose a `configureReadingLesson(content)` method that enters text content in the reading editor and saves the lesson.
3. THE Lesson_Editor_Page SHALL expose a `configureQuizLesson(question, options, correctIndex)` method that adds a question, adds all options, marks the correct answer, and saves the lesson.
4. THE Lesson_Editor_Page SHALL expose a `configureCodingLabLesson(starterCode, instructions)` method that enters starter code and instructions and saves the lesson.
5. THE Lesson_Editor_Page SHALL expose locator accessors for the video URL input, reading content editor, quiz question input, and code editor.
6. IF a lesson type's editor is not visible within the configured action timeout, THEN THE Lesson_Editor_Page SHALL surface the Playwright timeout error without swallowing it.

---

### Requirement 8: Authentication Test Scenarios

**User Story:** As a QA engineer, I want automated admin authentication tests, so that login behaviour is continuously verified.

#### Acceptance Criteria

1. WHEN Valid_Admin_Credentials are submitted on the Admin_Login_Page, THE Test_Suite SHALL assert that the application navigates to the admin dashboard.
2. WHEN Valid_Admin_Credentials are submitted, THE Test_Suite SHALL assert that the sidebar navigation is visible on the resulting page.
3. WHEN Valid_Admin_Credentials are submitted, THE Test_Suite SHALL assert that the user identity display is visible in the top-right area of the dashboard.
4. WHEN invalid credentials are submitted on the Admin_Login_Page, THE Test_Suite SHALL assert that an error message is visible.
5. THE Test_Suite SHALL use a `test.beforeEach()` hook in the Authentication describe block to navigate to the admin login page before each test.

---

### Requirement 9: Course Management Test Scenarios

**User Story:** As a QA engineer, I want automated course management tests, so that course creation and listing behaviour is continuously verified.

#### Acceptance Criteria

1. WHEN the "Courses" sidebar link is clicked from the dashboard, THE Test_Suite SHALL assert that the URL contains `/admin/courses`.
2. WHEN the Courses page is loaded, THE Test_Suite SHALL assert that the course list container is visible.
3. WHEN the Courses page is loaded, THE Test_Suite SHALL assert that the "New Course" button is visible.
4. WHEN a new course is created with a Dynamic_Course_Name and all required fields, THE Test_Suite SHALL assert that the application redirects to the curriculum page or the course appears in the course listing.
5. WHEN the course listing is searched for the Dynamic_Course_Name, THE Test_Suite SHALL assert that a course card with a matching title is visible.
6. THE Test_Suite SHALL use a `login()` helper in `test.beforeEach()` of the Course Management describe block to ensure the admin is authenticated before each test.

---

### Requirement 10: Curriculum Management Test Scenarios

**User Story:** As a QA engineer, I want automated curriculum management tests, so that module and lesson creation behaviour is continuously verified.

#### Acceptance Criteria

1. WHEN the edit icon is clicked on a course card, THE Test_Suite SHALL assert that the URL contains `/admin/courses/` and `/edit`.
2. WHEN the edit page loads, THE Test_Suite SHALL assert that the curriculum map container is visible.
3. WHEN a new module is added via `addModule()`, THE Test_Suite SHALL assert that the module name appears in the curriculum sidebar.
4. WHEN a new lesson is added via `addLesson()`, THE Test_Suite SHALL assert that the lesson name appears in the curriculum sidebar under the correct module.
5. WHEN a Video lesson is configured with a URL, THE Test_Suite SHALL assert that the video section is visible and the URL is accepted without a validation error.
6. WHEN a Reading lesson is configured with text content, THE Test_Suite SHALL assert that the content persists after a page reload.
7. WHEN a Quiz lesson is configured with a question, options, and a correct answer, THE Test_Suite SHALL assert that the question appears in the quiz UI after saving.
8. WHEN a Coding Lab lesson is configured with starter code and instructions, THE Test_Suite SHALL assert that the code editor is visible and the content is saved.
9. WHERE a lesson type feature is unstable or not yet functional, THE Test_Suite SHALL mark the affected test with `test.fixme()` and include a comment describing the known issue.
10. THE Test_Suite SHALL use a `createCourse()` helper in `test.beforeEach()` of the Curriculum Management describe block to ensure a fresh course exists before each curriculum test.

---

### Requirement 11: Reusable Helper Functions

**User Story:** As a QA engineer, I want shared helper functions, so that common multi-step flows are not duplicated across test files.

#### Acceptance Criteria

1. THE Test_Suite SHALL expose a `login(page, email, password)` helper that navigates to the admin login page, fills credentials, submits, and waits for the dashboard to load.
2. THE Test_Suite SHALL expose a `createCourse(page, courseDetails)` helper that logs in, navigates to the Courses page, clicks "New Course", fills all course fields, and saves.
3. THE Test_Suite SHALL expose an `addModule(page, moduleName)` helper that clicks "Add Module" on the curriculum page, enters the module name, and confirms.
4. THE Test_Suite SHALL expose an `addLesson(page, moduleName, lessonName, lessonType)` helper that clicks "Add Lesson" under the specified module, fills the lesson details, and confirms creation.
5. THE Test_Suite SHALL store helper functions in a dedicated `helpers/` or `utils/` directory, separate from page objects and test specs.

---

### Requirement 12: Assertions and Reporting

**User Story:** As a QA engineer, I want meaningful assertions and clear test descriptions, so that failures are easy to diagnose.

#### Acceptance Criteria

1. THE Test_Suite SHALL use `expect()` from `@playwright/test` for all assertions.
2. WHEN a test fails, THE Test_Suite SHALL capture a screenshot and attach it to the test report.
3. THE Test_Suite SHALL use descriptive strings in `test()` calls that clearly state the scenario being verified.
4. WHERE Playwright's built-in HTML reporter is available, THE Test_Suite SHALL be configured to generate an HTML report on test completion.
5. THE Test_Suite SHALL add a meaningful assertion after each major action (navigation, form submission, element interaction).
