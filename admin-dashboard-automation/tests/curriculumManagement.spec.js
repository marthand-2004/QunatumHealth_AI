// curriculumManagement.spec.js
// Test suite for the Owlet Campus admin curriculum management flows.
// Covers: edit page navigation, curriculum map visibility, module/lesson
// creation, and all four lesson type configurations.
// Requirements: 10.1–10.10, 12.1, 12.3, 12.5, 1.7

const { test, expect } = require('@playwright/test');
const { CoursesPage } = require('../pages/CoursesPage');
const { CurriculumPage } = require('../pages/CurriculumPage');
const { LessonEditorPage } = require('../pages/LessonEditorPage');
const { login, createCourse, addModule, addLesson } = require('../helpers/adminHelpers');

// ---------------------------------------------------------------------------
// Admin credentials — read from environment variables for CI safety.
// ---------------------------------------------------------------------------
const ADMIN_EMAIL    = process.env.ADMIN_EMAIL    || 'raghuram@gmail.com';
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'Ruaf@1489';

// Default course details — title is overridden per test with a unique value
const BASE_COURSE_DETAILS = {
  tagline:     'Curriculum test course',
  description: 'Created by Playwright automation for curriculum testing.',
  price:       '0',
  duration:    '1',
  category:    'Technology',
  difficulty:  'Beginner',
};

test.describe('Curriculum Management', () => {
  // -------------------------------------------------------------------------
  // beforeEach — Requirement 10.10
  // Log in and create a fresh course before every curriculum test.
  // Each test gets its own unique course to stay independent.
  // -------------------------------------------------------------------------
  test.beforeEach(async ({ page }) => {
    await login(page, ADMIN_EMAIL, ADMIN_PASSWORD);
    await createCourse(page, {
      ...BASE_COURSE_DETAILS,
      title: `Curriculum Course ${Date.now()}`,
    });
  });

  // -------------------------------------------------------------------------
  // Test 1: Edit page URL is correct — Requirement 10.1
  // Navigates to the courses listing, clicks edit on the created course,
  // and asserts the URL contains /admin/courses/{id}/edit.
  // -------------------------------------------------------------------------
  test('should navigate to edit page when edit icon is clicked', async ({ page }) => {
    const coursesPage = new CoursesPage(page);

    // Navigate to the courses listing
    await coursesPage.navigate();

    // Click the edit icon on the first course card (the one just created)
    // Uses the first available edit button since we just created this course
    await page.getByRole('button', { name: /edit/i }).first().click();

    // Assert the URL contains both /admin/courses/ and /edit
    await expect(page).toHaveURL(/\/admin\/courses\/.+\/edit/);
  });

  // -------------------------------------------------------------------------
  // Test 2: Curriculum map is visible on edit page — Requirement 10.2
  // Asserts the curriculum builder container renders after navigating to edit.
  // -------------------------------------------------------------------------
  test('should show curriculum map on edit page', async ({ page }) => {
    const curriculumPage = new CurriculumPage(page);

    // The beforeEach already navigated to the curriculum page after course creation
    // Assert the curriculum map container is visible
    await expect(curriculumPage.curriculumMap).toBeVisible();
  });

  // -------------------------------------------------------------------------
  // Test 3: Add a module — Requirement 10.3
  // Adds a new module and asserts it appears in the curriculum sidebar.
  // -------------------------------------------------------------------------
  test('should show new module in curriculum sidebar after adding', async ({ page }) => {
    const curriculumPage = new CurriculumPage(page);
    const moduleName = `Module ${Date.now()}`;

    // Add a new module using the helper
    await addModule(page, moduleName);

    // Assert the module name is visible in the curriculum sidebar
    await expect(curriculumPage.getModuleLocator(moduleName)).toBeVisible();
  });

  // -------------------------------------------------------------------------
  // Test 4: Add a lesson — Requirement 10.4
  // Adds a module, then a lesson under it, and asserts the lesson appears.
  // -------------------------------------------------------------------------
  test('should show new lesson in curriculum sidebar after adding', async ({ page }) => {
    const curriculumPage = new CurriculumPage(page);
    const moduleName = `Module ${Date.now()}`;
    const lessonName = `Lesson ${Date.now()}`;

    // First add a module to add the lesson under
    await addModule(page, moduleName);

    // Add a Video lesson under the module
    await addLesson(page, moduleName, lessonName, 'Video');

    // Assert the lesson name is visible in the curriculum sidebar
    await expect(curriculumPage.getLessonLocator(lessonName)).toBeVisible();
  });

  // -------------------------------------------------------------------------
  // Test 5A: Configure a Video lesson — Requirement 10.5
  // Adds a video lesson, enters a URL, and asserts it's accepted without error.
  // -------------------------------------------------------------------------
  test('should configure Video lesson with URL without validation error', async ({ page }) => {
    const lessonEditor = new LessonEditorPage(page);
    const moduleName = `Video Module ${Date.now()}`;
    const lessonName = `Video Lesson ${Date.now()}`;

    // Add module and video lesson
    await addModule(page, moduleName);
    await addLesson(page, moduleName, lessonName, 'Video');

    // Click the lesson to open the editor
    await page.getByText(lessonName).click();

    // Configure the video lesson with a sample URL
    await lessonEditor.configureVideoLesson('https://www.youtube.com/watch?v=dQw4w9WgXcQ');

    // Assert no validation error is visible
    const validationError = page.getByRole('alert').or(page.getByText(/invalid url|error/i)).first();
    const hasError = await validationError.count() > 0 && await validationError.isVisible();
    expect(hasError).toBe(false);

    // Assert the video section is visible
    await expect(lessonEditor.videoUrlInput).toBeVisible();
  });

  // -------------------------------------------------------------------------
  // Test 5B: Configure a Reading lesson — Requirement 10.6
  // Adds a reading lesson, enters content, saves, reloads, and checks persistence.
  // -------------------------------------------------------------------------
  test('should persist Reading lesson content after page reload', async ({ page }) => {
    const lessonEditor = new LessonEditorPage(page);
    const moduleName = `Reading Module ${Date.now()}`;
    const lessonName = `Reading Lesson ${Date.now()}`;
    const content = 'This is the reading lesson content for automation testing.';

    // Add module and reading lesson
    await addModule(page, moduleName);
    await addLesson(page, moduleName, lessonName, 'Reading');

    // Click the lesson to open the editor
    await page.getByText(lessonName).click();

    // Configure the reading lesson with content
    await lessonEditor.configureReadingLesson(content);

    // Reload the page to verify content persistence
    await page.reload();

    // Assert the content is still visible after reload
    await expect(page.getByText(content, { exact: false })).toBeVisible();
  });

  // -------------------------------------------------------------------------
  // Test 5C: Configure a Quiz lesson — Requirement 10.7
  // Adds a quiz lesson, adds a question with options, marks correct answer, saves.
  // test.fixme() applied if the quiz editor is not yet stable.
  // -------------------------------------------------------------------------
  test('should show Quiz question in UI after saving', async ({ page }) => {
    // test.fixme() — uncomment if the quiz editor is unstable in the current build
    // test.fixme(true, 'Quiz editor is not yet stable — pending backend implementation');

    const lessonEditor = new LessonEditorPage(page);
    const moduleName = `Quiz Module ${Date.now()}`;
    const lessonName = `Quiz Lesson ${Date.now()}`;
    const question = 'What is Playwright?';
    const options = ['A test framework', 'A browser', 'A database', 'A language'];

    // Add module and quiz lesson
    await addModule(page, moduleName);
    await addLesson(page, moduleName, lessonName, 'Quiz');

    // Click the lesson to open the editor
    await page.getByText(lessonName).click();

    // Configure the quiz with a question, options, and correct answer (index 0)
    await lessonEditor.configureQuizLesson(question, options, 0);

    // Assert the question text appears in the quiz UI after saving
    await expect(page.getByText(question, { exact: false })).toBeVisible();
  });

  // -------------------------------------------------------------------------
  // Test 5D: Configure a Coding Lab lesson — Requirement 10.8
  // Adds a coding lab lesson, enters starter code, saves, and checks the editor.
  // test.fixme() applied if the code editor is not yet stable.
  // -------------------------------------------------------------------------
  test('should show code editor and save Coding Lab content', async ({ page }) => {
    // test.fixme() — uncomment if the coding lab editor is unstable
    // test.fixme(true, 'Coding Lab editor may not be available in all environments');

    const lessonEditor = new LessonEditorPage(page);
    const moduleName = `Coding Module ${Date.now()}`;
    const lessonName = `Coding Lesson ${Date.now()}`;
    const starterCode = 'console.log("Hello, World!");';
    const instructions = 'Print a greeting to the console.';

    // Add module and coding lab lesson
    await addModule(page, moduleName);
    await addLesson(page, moduleName, lessonName, 'Coding Lab');

    // Click the lesson to open the editor
    await page.getByText(lessonName).click();

    // Assert the code editor is visible before configuring
    await expect(lessonEditor.codeEditor).toBeVisible();

    // Configure the coding lab with starter code and instructions
    await lessonEditor.configureCodingLabLesson(starterCode, instructions);

    // Assert the code editor is still visible after saving (no crash/redirect)
    await expect(lessonEditor.codeEditor).toBeVisible();
  });
});
