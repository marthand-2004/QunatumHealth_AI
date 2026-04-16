# Design Document: Admin Dashboard Automation

## Overview

This design covers a Playwright + JavaScript test suite for the Admin Dashboard of the Owlet Campus web application (`https://owlet-campus.com/admin`). The suite follows the Page Object Model (POM) pattern to encapsulate selectors and interactions, keeping test logic clean and maintainable.

The suite is organised into three `test.describe()` blocks — **Authentication**, **Course Management**, and **Curriculum Management** — and uses shared helper functions to avoid duplication across test cases. Dynamic test data (timestamp-based course names) prevents collisions between runs.

Primary goals:
- Convert the manual admin test flow into automated, repeatable scripts
- Cover login, course creation, curriculum management, and all four lesson types
- Produce clear failure messages, screenshots on failure, and an HTML report

## Architecture

The project is a standalone Node.js package inside an `admin-dashboard-automation/` directory. Playwright is the primary runtime dependency; fast-check is added for property-based tests.

### Project Folder Structure

```
admin-dashboard-automation/
├── playwright.config.js          # Base URL, reporter, screenshot, browser config
├── package.json
├── pages/
│   ├── AdminLoginPage.js         # Page Object for /admin/login
│   ├── DashboardPage.js          # Page Object for the authenticated dashboard
│   ├── CoursesPage.js            # Page Object for /admin/courses
│   ├── CourseFormPage.js         # Page Object for the new/edit course form
│   ├── CurriculumPage.js         # Page Object for /admin/courses/{id}/edit
│   └── LessonEditorPage.js       # Page Object for lesson content configuration
├── helpers/
│   └── adminHelpers.js           # Shared helper functions: login, createCourse, addModule, addLesson
└── tests/
    ├── authentication.spec.js    # Describe block: Authentication
    ├── courseManagement.spec.js  # Describe block: Course Management
    └── curriculumManagement.spec.js  # Describe block: Curriculum Management
```

### Flow Diagram

```mermaid
flowchart TD
    A[Test Runner] --> B[playwright.config.js]
    B --> C{Test Specs}
    C --> D[authentication.spec.js]
    C --> E[courseManagement.spec.js]
    C --> F[curriculumManagement.spec.js]

    D --> G[AdminLoginPage]
    D --> H[DashboardPage]

    E --> I[adminHelpers.login]
    E --> J[CoursesPage]
    E --> K[CourseFormPage]

    F --> L[adminHelpers.createCourse]
    F --> M[CurriculumPage]
    F --> N[LessonEditorPage]

    G --> O[https://owlet-campus.com/admin/login]
    H --> O
    J --> P[https://owlet-campus.com/admin/courses]
    K --> P
    M --> Q[https://owlet-campus.com/admin/courses/{id}/edit]
    N --> Q

    O --> R[Assertions + HTML Report]
    P --> R
    Q --> R
```

## Components and Interfaces

### `playwright.config.js`

Configures the test runner globally.

| Setting | Value |
|---|---|
| `baseURL` | `https://owlet-campus.com/` |
| `testDir` | `./tests` |
| `timeout` | `60000` |
| `use.headless` | `true` |
| `use.screenshot` | `'only-on-failure'` |
| `use.navigationTimeout` | `30000` |
| `use.actionTimeout` | `15000` |
| `reporter` | `[['html', { open: 'never' }]]` |

### `AdminLoginPage` (pages/AdminLoginPage.js)

Encapsulates all interactions with the admin login form at `/admin/login`.

```
class AdminLoginPage
  constructor(page)

  // Locator accessors
  get emailInput()        → Locator   // getByLabel or getByPlaceholder for Email field
  get passwordInput()     → Locator   // getByLabel or getByPlaceholder for Password field
  get loginButton()       → Locator   // getByRole('button', { name: /login/i })

  // Methods
  navigate()                          → Promise<void>   // goto('/admin/login')
  fillForm(email, password)           → Promise<void>   // fills email + password fields
  submit()                            → Promise<void>   // clicks loginButton
  login(email, password)              → Promise<void>   // fillForm + submit
```

### `DashboardPage` (pages/DashboardPage.js)

Encapsulates post-login dashboard state and sidebar navigation.

```
class DashboardPage
  constructor(page)

  // Locator accessors
  get sidebarNav()        → Locator   // getByRole('navigation') or sidebar container
  get userIdentity()      → Locator   // top-right user display element

  // Methods
  waitForLoad()                       → Promise<void>   // waits for sidebarNav to be visible
  navigateTo(sectionName)             → Promise<void>   // clicks sidebar link by visible text
```

### `CoursesPage` (pages/CoursesPage.js)

Encapsulates the course listing at `/admin/courses`.

```
class CoursesPage
  constructor(page)

  // Locator accessors
  get courseList()        → Locator   // course list container element
  get newCourseButton()   → Locator   // getByRole('button', { name: /new course/i }) or link

  // Methods
  navigate()                          → Promise<void>   // goto('/admin/courses')
  clickNewCourse()                    → Promise<void>   // clicks newCourseButton
  searchCourse(title)                 → Locator         // returns locator for course card by title text
  clickEditCourse(title)              → Promise<void>   // clicks edit icon on matching course card
```

### `CourseFormPage` (pages/CourseFormPage.js)

Encapsulates the new/edit course form.

```
class CourseFormPage
  constructor(page)

  // Locator accessors
  get titleInput()        → Locator
  get taglineInput()      → Locator
  get descriptionInput()  → Locator
  get priceInput()        → Locator
  get durationInput()     → Locator
  get categorySelect()    → Locator
  get difficultySelect()  → Locator
  get thumbnailInput()    → Locator   // file input for thumbnail upload

  // Methods
  fillCourseDetails(details)          → Promise<void>
    // details: { title, tagline, description, price, duration, category, difficulty }
  uploadThumbnail(filePath)           → Promise<void>   // setInputFiles on thumbnailInput
  saveAndBuildCurriculum()            → Promise<void>   // clicks "Save and Build Curriculum"
```

### `CurriculumPage` (pages/CurriculumPage.js)

Encapsulates the curriculum builder at `/admin/courses/{id}/edit`.

```
class CurriculumPage
  constructor(page)

  // Locator accessors
  get curriculumMap()     → Locator   // curriculum map container

  // Methods
  addModule(moduleName)               → Promise<void>
    // clicks "Add Module", enters moduleName, confirms
  addLesson(moduleName, lessonName, lessonType)  → Promise<void>
    // clicks "Add Lesson" under moduleName, fills lessonName, selects lessonType, clicks "Create & Configure"
  getModuleLocator(moduleName)        → Locator   // locator for module entry in sidebar
  getLessonLocator(lessonName)        → Locator   // locator for lesson entry in sidebar
```

### `LessonEditorPage` (pages/LessonEditorPage.js)

Encapsulates lesson content configuration for all four lesson types.

```
class LessonEditorPage
  constructor(page)

  // Locator accessors
  get videoUrlInput()     → Locator   // video URL input field
  get readingEditor()     → Locator   // reading content editor (rich text or textarea)
  get quizQuestionInput() → Locator   // quiz question text input
  get codeEditor()        → Locator   // code editor element (Monaco or CodeMirror)

  // Methods
  configureVideoLesson(videoUrl)                          → Promise<void>
    // enters videoUrl, saves lesson
  configureReadingLesson(content)                         → Promise<void>
    // enters content in reading editor, saves lesson
  configureQuizLesson(question, options, correctIndex)    → Promise<void>
    // adds question, adds all options, marks correct answer at correctIndex, saves
  configureCodingLabLesson(starterCode, instructions)     → Promise<void>
    // enters starterCode in code editor, enters instructions, saves
```

### `adminHelpers.js` (helpers/adminHelpers.js)

Shared utility functions used across all describe blocks.

```js
// Navigates to /admin/login, fills credentials, submits, waits for dashboard
async function login(page, email, password)

// Calls login(), navigates to /admin/courses, clicks New Course,
// fills all course fields from courseDetails, saves
async function createCourse(page, courseDetails)

// Clicks "Add Module" on the curriculum page, enters moduleName, confirms
async function addModule(page, moduleName)

// Clicks "Add Lesson" under moduleName, fills lessonName,
// selects lessonType, confirms creation
async function addLesson(page, moduleName, lessonName, lessonType)
```

## Data Models

There are no persistent data models — the suite operates against the live application. Test data is defined as constants or generated at runtime.

### Test Data Shapes

```js
// Admin credentials (stored as constants; move to env vars for CI)
const ADMIN_EMAIL    = 'raghuram@gmail.com';
const ADMIN_PASSWORD = 'Ruaf@1489';
const ADMIN_URL      = 'https://owlet-campus.com/admin/login';

// Dynamic course name — timestamp suffix prevents collisions between runs
const courseName = `Test Course ${Date.now()}`;

// Course details object passed to fillCourseDetails() and createCourse()
const courseDetails = {
  title:       `Test Course ${Date.now()}`,
  tagline:     'Automated test course',
  description: 'Created by Playwright automation',
  price:       '0',
  duration:    '1',
  category:    'Technology',      // must match a valid dropdown option
  difficulty:  'Beginner',        // must match a valid dropdown option
};

// Lesson types (valid values for lessonType parameter)
type LessonType = 'Video' | 'Reading' | 'Quiz' | 'Coding Lab';

// Quiz lesson data
const quizData = {
  question:     'What is Playwright?',
  options:      ['A test framework', 'A browser', 'A database', 'A language'],
  correctIndex: 0,
};

// Coding Lab lesson data
const codingLabData = {
  starterCode:  'console.log("Hello, World!");',
  instructions: 'Print a greeting to the console.',
};
```

### Dynamic Test Data Strategy

Course names are generated with `Date.now()` to ensure uniqueness across parallel or sequential runs:

```js
// In test.beforeEach or at the top of a describe block
const uniqueCourseName = `Automation Course ${Date.now()}`;
```

This avoids collisions when tests are run multiple times against the same environment without cleanup.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

The following properties were derived from the acceptance criteria prework analysis. Properties are tested using **fast-check** for JavaScript. Each property runs a minimum of 100 iterations.

**PBT Applicability Assessment:** This feature is a Playwright test suite targeting a live web application. Most acceptance criteria describe specific UI interactions (EXAMPLE or SMOKE). However, several criteria describe universal behaviors that hold across a range of inputs — particularly the fillForm round-trips, invalid credential handling, and content persistence patterns. These are suitable for property-based testing with fast-check generating the input data while Playwright drives the browser.

**Property Reflection:** After prework analysis, the following consolidations were made:
- 10.3 (module appears after addModule) subsumes 6.1 (addModule method behavior) — kept as one integration-level property.
- 10.4 (lesson appears after addLesson) subsumes 6.2 — kept as one integration-level property.
- 9.4 and 9.5 are combined: course creation and subsequent listing visibility are one round-trip property.
- 10.5, 10.6, 10.7, 10.8 (lesson type configurations) are combined into one property covering all lesson types.
- 11.2, 11.3, 11.4 (helper functions) are subsumed by the corresponding integration properties.
- 6.3 and 6.4 (locator methods) are combined into one property.

### Property 1: AdminLoginPage fillForm round-trip

*For any* email string and password string, after calling `fillForm(email, password)` on a navigated `AdminLoginPage`, the email input field's value should equal the email argument and the password input field's value should equal the password argument.

**Validates: Requirements 2.2**

### Property 2: Invalid credentials always produce an authentication error

*For any* email/password pair that is not the registered admin account, submitting the admin login form with those credentials should result in a visible error message on the page.

**Validates: Requirements 8.4**

### Property 3: CourseFormPage fillCourseDetails round-trip

*For any* course details object containing non-empty strings for title, tagline, description, price, and duration, after calling `fillCourseDetails(details)` on a loaded `CourseFormPage`, each corresponding text input field's value should equal the value from the details object.

**Validates: Requirements 5.1**

### Property 4: Course creation and listing round-trip

*For any* valid course details object with a unique title, after calling `createCourse()`, the course title should appear as a visible course card in the `/admin/courses` listing.

**Validates: Requirements 9.4, 9.5**

### Property 5: Module addition appears in curriculum sidebar

*For any* non-empty module name string, after calling `addModule(moduleName)` on the curriculum page, the module name should appear as a visible entry in the curriculum sidebar.

**Validates: Requirements 10.3, 6.1**

### Property 6: Lesson addition appears in curriculum sidebar

*For any* lesson name string and valid lesson type (`'Video'`, `'Reading'`, `'Quiz'`, `'Coding Lab'`), after calling `addLesson(moduleName, lessonName, lessonType)`, the lesson name should appear as a visible entry in the curriculum sidebar under the correct module.

**Validates: Requirements 10.4, 6.2**

### Property 7: Lesson content configuration persists after save

*For any* lesson type and its corresponding content input (video URL, reading text, quiz question, or starter code), after configuring the lesson and saving, the configured content should be visible in the lesson editor without a validation error.

**Validates: Requirements 10.5, 10.6, 10.7, 10.8**

### Property 8: Curriculum locator methods resolve to visible elements

*For any* module name or lesson name that has been added to the curriculum, calling `getModuleLocator(moduleName)` or `getLessonLocator(lessonName)` should return a locator that resolves to a visible element in the curriculum sidebar.

**Validates: Requirements 6.3, 6.4**

## Error Handling

| Scenario | Handling |
|---|---|
| Playwright locator not found | Playwright throws `TimeoutError` by default; page objects do not catch or suppress it (Requirements 2.6, 3.5, 5.5, 6.6, 7.6) |
| Network / navigation timeout | Playwright's configured `navigationTimeout: 30000` applies; tests fail with a descriptive timeout message |
| Test data collision (duplicate course name) | Timestamp suffix in `Dynamic_Course_Name` prevents collisions; each run generates a unique name |
| Invalid credentials test | Negative test uses a generated random email/password pair that is guaranteed not to be the valid admin account |
| Unstable lesson type features | Affected tests are marked with `test.fixme()` and include a comment describing the known issue (Requirement 10.9) |
| Missing env vars for CI credentials | Tests read `process.env.ADMIN_EMAIL` / `process.env.ADMIN_PASSWORD`; if undefined, fall back to the constants defined in the test file |
| File upload for thumbnail | `uploadThumbnail()` uses `page.setInputFiles()` with a bundled test fixture image; if the file is missing, Playwright throws a descriptive error |

## Testing Strategy

### Dual Testing Approach

The suite uses two complementary layers:

1. **Example-based tests** (Playwright `test()` blocks) — cover specific, deterministic scenarios: successful login, course listing visibility, curriculum navigation, and each lesson type configuration.
2. **Property-based tests** (fast-check + Playwright) — cover universal properties across generated inputs: fillForm round-trips, invalid credential handling, course creation round-trips, and content persistence.

Both layers are necessary: example tests catch concrete bugs in specific flows; property tests verify general correctness across the input space.

### Property-Based Testing Setup

- Library: **fast-check** (`npm install --save-dev fast-check`)
- Minimum iterations per property: **100** (`{ numRuns: 100 }`)
- Each property test is tagged with a comment referencing the design property:
  ```js
  // Feature: admin-dashboard-automation, Property 1: AdminLoginPage fillForm round-trip
  ```

### Test File Breakdown

**`tests/authentication.spec.js`** — Describe block: "Authentication"

| Test | Type | Requirement |
|---|---|---|
| `should navigate to admin login page` | Example | 2.1, 8.5 |
| `should login successfully with valid credentials` | Example | 8.1 |
| `should show sidebar navigation after login` | Example | 8.2 |
| `should show user identity display after login` | Example | 8.3 |
| `should show error for invalid credentials (property)` | Property | 8.4 |
| `fillForm should populate email and password fields (property)` | Property | 2.2 |

**`tests/courseManagement.spec.js`** — Describe block: "Course Management"

| Test | Type | Requirement |
|---|---|---|
| `should navigate to courses page via sidebar` | Example | 9.1 |
| `should show course list container on courses page` | Example | 9.2 |
| `should show New Course button on courses page` | Example | 9.3 |
| `fillCourseDetails should populate all form fields (property)` | Property | 5.1 |
| `should create a new course and redirect to curriculum (property)` | Property | 9.4 |
| `should show created course in listing (property)` | Property | 9.5 |

**`tests/curriculumManagement.spec.js`** — Describe block: "Curriculum Management"

| Test | Type | Requirement |
|---|---|---|
| `should navigate to edit page when edit icon is clicked` | Example | 10.1 |
| `should show curriculum map on edit page` | Example | 10.2 |
| `should show module in sidebar after addModule (property)` | Property | 10.3 |
| `should show lesson in sidebar after addLesson (property)` | Property | 10.4 |
| `should configure Video lesson without validation error (property)` | Property | 10.5 |
| `should persist Reading lesson content after reload (property)` | Property | 10.6 |
| `should show Quiz question in UI after save (property)` | Property | 10.7 |
| `should show code editor and save Coding Lab content (property)` | Property | 10.8 |
| `curriculum locator methods should resolve to visible elements (property)` | Property | 6.3, 6.4 |

### `test.beforeEach` Hooks

```js
// authentication.spec.js
test.beforeEach(async ({ page }) => {
  const loginPage = new AdminLoginPage(page);
  await loginPage.navigate();
});

// courseManagement.spec.js
test.beforeEach(async ({ page }) => {
  await login(page, ADMIN_EMAIL, ADMIN_PASSWORD);
});

// curriculumManagement.spec.js
test.beforeEach(async ({ page }) => {
  await createCourse(page, { ...courseDetails, title: `Test Course ${Date.now()}` });
});
```

### Screenshot and Reporting Configuration

```js
// playwright.config.js
use: {
  screenshot: 'only-on-failure',  // Requirement 12.2 — captures screenshot on failure
},
reporter: [['html', { open: 'never' }]],  // Requirement 12.4 — HTML report, CI-safe
```

Screenshots are automatically attached to the HTML report when a test fails. No additional configuration is needed.

### Run Commands

```bash
# Install dependencies
npm install

# Run all tests (single execution, no watch mode)
npx playwright test

# Run a specific spec file
npx playwright test tests/authentication.spec.js

# View the HTML report
npx playwright show-report
```
