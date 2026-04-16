// courseManagement.spec.js
// Test suite for the Owlet Campus admin course management flows.
// Covers: courses page navigation, course list visibility, New Course button,
// course creation, and course listing verification.
// Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 12.1, 12.3, 12.5, 1.7

const { test, expect } = require('@playwright/test');
const { CoursesPage } = require('../pages/CoursesPage');
const { CourseFormPage } = require('../pages/CourseFormPage');
const { DashboardPage } = require('../pages/DashboardPage');
const { login } = require('../helpers/adminHelpers');

// ---------------------------------------------------------------------------
// Admin credentials — read from environment variables for CI safety.
// Falls back to hardcoded constants for local development runs.
// ---------------------------------------------------------------------------
const ADMIN_EMAIL    = process.env.ADMIN_EMAIL    || 'raghuram@gmail.com';
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'Ruaf@1489';

// ---------------------------------------------------------------------------
// Dynamic course name — timestamp suffix prevents collisions between runs.
// Declared at describe scope so all tests in this block share the same name.
// ---------------------------------------------------------------------------
const COURSE_NAME = `Test Course ${Date.now()}`;

// Default course details used for course creation tests
const COURSE_DETAILS = {
  title:       COURSE_NAME,
  tagline:     'Automated test course tagline',
  description: 'This course was created by Playwright automation.',
  price:       '0',
  duration:    '1',
  category:    'Technology',
  difficulty:  'Beginner',
};

test.describe('Course Management', () => {
  // -------------------------------------------------------------------------
  // beforeEach — Requirement 9.6
  // Log in as admin before every test in this describe block.
  // -------------------------------------------------------------------------
  test.beforeEach(async ({ page }) => {
    await login(page, ADMIN_EMAIL, ADMIN_PASSWORD);
  });

  // -------------------------------------------------------------------------
  // Test 1: Navigate to Courses page via sidebar — Requirement 9.1
  // Clicks "Courses" in the sidebar and asserts the URL updates.
  // -------------------------------------------------------------------------
  test('should navigate to courses page via sidebar', async ({ page }) => {
    const dashboard = new DashboardPage(page);

    // Click the "Courses" link in the sidebar navigation
    await dashboard.navigateTo('Courses');

    // Assert the URL now contains /admin/courses
    await expect(page).toHaveURL(/\/admin\/courses/);
  });

  // -------------------------------------------------------------------------
  // Test 2: Course list container is visible — Requirement 9.2
  // Navigates directly to /admin/courses and asserts the list is rendered.
  // -------------------------------------------------------------------------
  test('should show course list container on courses page', async ({ page }) => {
    const coursesPage = new CoursesPage(page);

    // Navigate directly to the courses listing page
    await coursesPage.navigate();

    // Assert the course list container is visible
    await expect(coursesPage.courseList).toBeVisible();
  });

  // -------------------------------------------------------------------------
  // Test 3: "New Course" button is visible — Requirement 9.3
  // Navigates to /admin/courses and asserts the New Course button is present.
  // -------------------------------------------------------------------------
  test('should show New Course button on courses page', async ({ page }) => {
    const coursesPage = new CoursesPage(page);

    // Navigate to the courses listing page
    await coursesPage.navigate();

    // Assert the "New Course" button/link is visible
    await expect(coursesPage.newCourseButton).toBeVisible();
  });

  // -------------------------------------------------------------------------
  // Test 4: Create a new course — Requirement 9.4
  // Fills the course form and asserts a redirect to curriculum or listing.
  // -------------------------------------------------------------------------
  test('should create a new course and redirect to curriculum or listing', async ({ page }) => {
    const coursesPage = new CoursesPage(page);
    const courseForm = new CourseFormPage(page);

    // Navigate to courses and click "New Course"
    await coursesPage.navigate();
    await coursesPage.clickNewCourse();

    // Fill all course details with the dynamic course name
    await courseForm.fillCourseDetails(COURSE_DETAILS);

    // Click "Save and Build Curriculum"
    await courseForm.saveAndBuildCurriculum();

    // Assert: either redirected to curriculum page OR back to course listing
    // Both are valid outcomes per Requirement 9.4
    const onCurriculum = page.url().includes('/edit');
    const onListing    = page.url().includes('/admin/courses');
    expect(onCurriculum || onListing).toBe(true);
  });

  // -------------------------------------------------------------------------
  // Test 5: Created course appears in listing — Requirement 9.5
  // Creates a course then navigates to /admin/courses to verify it's listed.
  // -------------------------------------------------------------------------
  test('should show created course in the course listing', async ({ page }) => {
    const coursesPage = new CoursesPage(page);
    const courseForm = new CourseFormPage(page);

    // Use a unique name for this specific test to avoid cross-test interference
    const uniqueName = `Verify Course ${Date.now()}`;

    // Navigate to courses and create a new course
    await coursesPage.navigate();
    await coursesPage.clickNewCourse();
    await courseForm.fillCourseDetails({ ...COURSE_DETAILS, title: uniqueName });
    await courseForm.saveAndBuildCurriculum();

    // Navigate back to the courses listing
    await coursesPage.navigate();

    // Assert the course card with the matching title is visible
    const courseCard = coursesPage.searchCourse(uniqueName);
    await expect(courseCard).toBeVisible();
  });
});
