# Implementation Plan: Admin Dashboard Automation

## Overview

Build a standalone Playwright + JavaScript test suite for the Owlet Campus admin panel following the Page Object Model pattern. Tasks progress from project scaffolding through page object implementations, helper functions, and test specs, finishing with property-based tests using fast-check.

## Tasks

- [x] 1. Scaffold project structure and configuration
  - Create `admin-dashboard-automation/` directory with `package.json` declaring `@playwright/test` and `fast-check` as dev dependencies
  - Create `playwright.config.js` with `baseURL: 'https://owlet-campus.com/'`, `testDir: './tests'`, `timeout: 60000`, `use.headless: true`, `use.screenshot: 'only-on-failure'`, `use.navigationTimeout: 30000`, `use.actionTimeout: 15000`, and `reporter: [['html', { open: 'never' }]]`
  - Create empty placeholder files: `pages/.gitkeep`, `helpers/.gitkeep`, `tests/.gitkeep`
  - Add inline comments in `playwright.config.js` explaining each configuration block
  - _Requirements: 1.1, 1.3, 1.5, 12.4_

- [x] 2. Implement `AdminLoginPage` page object
  - [x] 2.1 Create `pages/AdminLoginPage.js` with constructor, locator accessors (`emailInput`, `passwordInput`, `loginButton`), and methods `navigate()`, `fillForm(email, password)`, `submit()`, `login(email, password)`
    - Use `getByLabel`, `getByPlaceholder`, or `getByRole` for all locators — no CSS selectors
    - `navigate()` calls `page.goto('/admin/login')`
    - `login()` calls `fillForm` then `submit` as a convenience wrapper
    - Add inline comments for each method block
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 1.4, 1.6_

  - [ ]* 2.2 Write property test for `AdminLoginPage.fillForm` round-trip (Property 1)
    - **Property 1: AdminLoginPage fillForm round-trip**
    - Use `fc.string()` arbitraries for email and password
    - After `fillForm(email, password)`, assert `emailInput.inputValue()` equals email and `passwordInput.inputValue()` equals password
    - Run with `{ numRuns: 100 }`
    - Tag with comment: `// Feature: admin-dashboard-automation, Property 1: AdminLoginPage fillForm round-trip`
    - **Validates: Requirements 2.2**

- [x] 3. Implement `DashboardPage` page object
  - [x] 3.1 Create `pages/DashboardPage.js` with constructor, locator accessors (`sidebarNav`, `userIdentity`), and methods `waitForLoad()`, `navigateTo(sectionName)`
    - `waitForLoad()` waits for `sidebarNav` to be visible
    - `navigateTo(sectionName)` clicks the sidebar link matching the given text
    - Errors from Playwright locators must propagate — do not catch or suppress them
    - Add inline comments for each method block
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 1.4, 1.6_

- [x] 4. Implement `CoursesPage` page object
  - [x] 4.1 Create `pages/CoursesPage.js` with constructor, locator accessors (`courseList`, `newCourseButton`), and methods `navigate()`, `clickNewCourse()`, `searchCourse(title)`, `clickEditCourse(title)`
    - `navigate()` calls `page.goto('/admin/courses')`
    - `searchCourse(title)` returns a locator for the course card matching the title text
    - `clickEditCourse(title)` clicks the edit icon on the matching course card
    - Use semantic locators throughout
    - Add inline comments for each method block
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 1.4, 1.6_

- [x] 5. Implement `CourseFormPage` page object
  - [x] 5.1 Create `pages/CourseFormPage.js` with constructor, locator accessors for all form fields (`titleInput`, `taglineInput`, `descriptionInput`, `priceInput`, `durationInput`, `categorySelect`, `difficultySelect`, `thumbnailInput`), and methods `fillCourseDetails(details)`, `uploadThumbnail(filePath)`, `saveAndBuildCurriculum()`
    - `fillCourseDetails(details)` fills Title, Tagline, Description, Price, Duration, Category, and Difficulty from the details object
    - `uploadThumbnail(filePath)` uses `page.setInputFiles()` on the thumbnail input
    - `saveAndBuildCurriculum()` clicks the "Save and Build Curriculum" button
    - Errors from Playwright locators must propagate — do not catch or suppress them
    - Add inline comments for each method block
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 1.4, 1.6_

  - [ ]* 5.2 Write property test for `CourseFormPage.fillCourseDetails` round-trip (Property 3)
    - **Property 3: CourseFormPage fillCourseDetails round-trip**
    - Use `fc.record({ title: fc.string({ minLength: 1 }), tagline: fc.string({ minLength: 1 }), description: fc.string({ minLength: 1 }), price: fc.string({ minLength: 1 }), duration: fc.string({ minLength: 1 }) })` arbitrary
    - After `fillCourseDetails(details)`, assert each text input's `inputValue()` equals the corresponding field
    - Run with `{ numRuns: 100 }`
    - Tag with comment: `// Feature: admin-dashboard-automation, Property 3: CourseFormPage fillCourseDetails round-trip`
    - **Validates: Requirements 5.1**

- [x] 6. Implement `CurriculumPage` page object
  - [x] 6.1 Create `pages/CurriculumPage.js` with constructor, locator accessor (`curriculumMap`), and methods `addModule(moduleName)`, `addLesson(moduleName, lessonName, lessonType)`, `getModuleLocator(moduleName)`, `getLessonLocator(lessonName)`
    - `addModule(moduleName)` clicks "Add Module", enters the name, and confirms
    - `addLesson(moduleName, lessonName, lessonType)` clicks "Add Lesson" under the specified module, fills the lesson name, selects the lesson type, and clicks "Create & Configure"
    - `getModuleLocator` and `getLessonLocator` return Playwright locators for sidebar entries
    - Errors from Playwright locators must propagate — do not catch or suppress them
    - Add inline comments for each method block
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 1.4, 1.6_

- [x] 7. Implement `LessonEditorPage` page object
  - [x] 7.1 Create `pages/LessonEditorPage.js` with constructor, locator accessors (`videoUrlInput`, `readingEditor`, `quizQuestionInput`, `codeEditor`), and methods `configureVideoLesson(videoUrl)`, `configureReadingLesson(content)`, `configureQuizLesson(question, options, correctIndex)`, `configureCodingLabLesson(starterCode, instructions)`
    - Each configure method enters the relevant content and saves the lesson
    - `configureQuizLesson` adds all options and marks the correct answer at `correctIndex`
    - Errors from Playwright locators must propagate — do not catch or suppress them
    - Add inline comments for each method block
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 1.4, 1.6_

- [x] 8. Implement `adminHelpers.js` shared helper functions
  - [x] 8.1 Create `helpers/adminHelpers.js` exporting `login(page, email, password)`, `createCourse(page, courseDetails)`, `addModule(page, moduleName)`, `addLesson(page, moduleName, lessonName, lessonType)`
    - `login` uses `AdminLoginPage` to navigate, fill, and submit, then uses `DashboardPage.waitForLoad()` to confirm success
    - `createCourse` calls `login`, navigates to `CoursesPage`, clicks "New Course", uses `CourseFormPage.fillCourseDetails()`, and saves
    - `addModule` and `addLesson` delegate to the corresponding `CurriculumPage` methods
    - Export all four functions as named exports
    - Add inline comments for each function
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 1.6_

- [x] 9. Checkpoint — wire page objects and helpers together
  - Ensure all page object files import correctly from each other where needed
  - Ensure `adminHelpers.js` imports `AdminLoginPage`, `DashboardPage`, `CoursesPage`, `CourseFormPage`, and `CurriculumPage`
  - Verify no circular imports exist
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement `tests/authentication.spec.js`
  - [x] 10.1 Create `tests/authentication.spec.js` with a `test.describe('Authentication', ...)` block containing a `test.beforeEach` that navigates to the admin login page using `AdminLoginPage.navigate()`
    - Define `ADMIN_EMAIL`, `ADMIN_PASSWORD` constants (with `process.env` fallback for CI)
    - Add example tests: `should navigate to admin login page`, `should login successfully with valid credentials`, `should show sidebar navigation after login`, `should show user identity display after login`
    - Each test uses `expect()` from `@playwright/test` for all assertions
    - Add descriptive strings to each `test()` call
    - Add inline comments for each logical block
    - _Requirements: 8.1, 8.2, 8.3, 8.5, 12.1, 12.3, 12.5, 1.7_

  - [ ]* 10.2 Write property test for invalid credentials (Property 2)
    - **Property 2: Invalid credentials always produce an authentication error**
    - Use `fc.record({ email: fc.emailAddress(), password: fc.string({ minLength: 1 }) })` filtered to exclude the valid admin credentials
    - For each generated pair, submit the login form and assert that an error message is visible
    - Run with `{ numRuns: 100 }`
    - Tag with comment: `// Feature: admin-dashboard-automation, Property 2: Invalid credentials always produce an authentication error`
    - **Validates: Requirements 8.4**

- [x] 11. Implement `tests/courseManagement.spec.js`
  - [x] 11.1 Create `tests/courseManagement.spec.js` with a `test.describe('Course Management', ...)` block containing a `test.beforeEach` that calls `login(page, ADMIN_EMAIL, ADMIN_PASSWORD)`
    - Add example tests: `should navigate to courses page via sidebar`, `should show course list container on courses page`, `should show New Course button on courses page`
    - Each test uses `expect()` for all assertions with descriptive test names
    - Add inline comments for each logical block
    - _Requirements: 9.1, 9.2, 9.3, 9.6, 12.1, 12.3, 12.5, 1.7_

  - [ ]* 11.2 Write property test for course creation and listing round-trip (Properties 3 & 4)
    - **Property 4: Course creation and listing round-trip**
    - Use `fc.record({ title: fc.string({ minLength: 1 }) })` combined with a timestamp suffix to ensure uniqueness
    - After `createCourse()`, assert the course title appears as a visible card in `/admin/courses`
    - Run with `{ numRuns: 100 }`
    - Tag with comment: `// Feature: admin-dashboard-automation, Property 4: Course creation and listing round-trip`
    - **Validates: Requirements 9.4, 9.5**

- [x] 12. Implement `tests/curriculumManagement.spec.js`
  - [x] 12.1 Create `tests/curriculumManagement.spec.js` with a `test.describe('Curriculum Management', ...)` block containing a `test.beforeEach` that calls `createCourse(page, { ...courseDetails, title: \`Test Course ${Date.now()}\` })`
    - Add example tests: `should navigate to edit page when edit icon is clicked`, `should show curriculum map on edit page`
    - Mark any unstable lesson type tests with `test.fixme()` and a comment describing the known issue
    - Each test uses `expect()` for all assertions with descriptive test names
    - Add inline comments for each logical block
    - _Requirements: 10.1, 10.2, 10.9, 10.10, 12.1, 12.3, 12.5, 1.7_

  - [ ]* 12.2 Write property test for module addition in curriculum sidebar (Property 5)
    - **Property 5: Module addition appears in curriculum sidebar**
    - Use `fc.string({ minLength: 1 })` for module name
    - After `addModule(moduleName)`, assert the module name is visible in the curriculum sidebar
    - Run with `{ numRuns: 100 }`
    - Tag with comment: `// Feature: admin-dashboard-automation, Property 5: Module addition appears in curriculum sidebar`
    - **Validates: Requirements 10.3, 6.1**

  - [ ]* 12.3 Write property test for lesson addition in curriculum sidebar (Property 6)
    - **Property 6: Lesson addition appears in curriculum sidebar**
    - Use `fc.string({ minLength: 1 })` for lesson name and `fc.constantFrom('Video', 'Reading', 'Quiz', 'Coding Lab')` for lesson type
    - After `addLesson(moduleName, lessonName, lessonType)`, assert the lesson name is visible under the correct module
    - Run with `{ numRuns: 100 }`
    - Tag with comment: `// Feature: admin-dashboard-automation, Property 6: Lesson addition appears in curriculum sidebar`
    - **Validates: Requirements 10.4, 6.2**

  - [ ]* 12.4 Write property test for lesson content configuration persistence (Property 7)
    - **Property 7: Lesson content configuration persists after save**
    - Use `fc.constantFrom('Video', 'Reading', 'Quiz', 'Coding Lab')` to drive lesson type selection; generate matching content with `fc.string({ minLength: 1 })`
    - After configuring and saving each lesson type, assert the content is visible without a validation error
    - Run with `{ numRuns: 100 }`
    - Tag with comment: `// Feature: admin-dashboard-automation, Property 7: Lesson content configuration persists after save`
    - **Validates: Requirements 10.5, 10.6, 10.7, 10.8**

  - [ ]* 12.5 Write property test for curriculum locator methods (Property 8)
    - **Property 8: Curriculum locator methods resolve to visible elements**
    - Use `fc.string({ minLength: 1 })` for module and lesson names
    - After adding a module and lesson, assert `getModuleLocator(moduleName)` and `getLessonLocator(lessonName)` each resolve to a visible element
    - Run with `{ numRuns: 100 }`
    - Tag with comment: `// Feature: admin-dashboard-automation, Property 8: Curriculum locator methods resolve to visible elements`
    - **Validates: Requirements 6.3, 6.4**

- [x] 13. Final checkpoint — Ensure all tests pass
  - Run `npx playwright test` and confirm all non-fixme tests pass
  - Verify the HTML report is generated at `playwright-report/index.html`
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests (Properties 1–8) validate universal correctness properties using fast-check with `{ numRuns: 100 }`
- Unit/example tests validate specific scenarios and edge cases
- Admin credentials should be stored as `process.env.ADMIN_EMAIL` / `process.env.ADMIN_PASSWORD` with constants as fallback for local runs
- Tests for unstable lesson type features should be marked with `test.fixme()` per Requirement 10.9
