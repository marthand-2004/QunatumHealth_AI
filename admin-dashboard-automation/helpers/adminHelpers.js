// adminHelpers.js
// Shared helper functions for the Owlet Campus admin dashboard test suite.
// These utilities encapsulate common multi-step flows so individual test cases
// stay concise and independent. Each helper composes the relevant Page Objects.
// Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 1.6

const { AdminLoginPage } = require('../pages/AdminLoginPage');
const { DashboardPage } = require('../pages/DashboardPage');
const { CoursesPage } = require('../pages/CoursesPage');
const { CourseFormPage } = require('../pages/CourseFormPage');
const { CurriculumPage } = require('../pages/CurriculumPage');

// ---------------------------------------------------------------------------
// login(page, email, password) — Requirement 11.1
// Navigates to the admin login page, fills credentials, submits the form,
// and waits for the dashboard sidebar to confirm a successful login.
//
// @param {import('@playwright/test').Page} page
// @param {string} email
// @param {string} password
// ---------------------------------------------------------------------------
async function login(page, email, password) {
  // Instantiate the login page object and navigate to /admin/login
  const loginPage = new AdminLoginPage(page);
  await loginPage.navigate();

  // Fill credentials and submit the login form
  await loginPage.login(email, password);

  // Wait for the dashboard to load — confirms login was successful
  const dashboard = new DashboardPage(page);
  await dashboard.waitForLoad();
}

// ---------------------------------------------------------------------------
// createCourse(page, courseDetails) — Requirement 11.2
// Logs in (if not already authenticated), navigates to the Courses page,
// clicks "New Course", fills all course fields, and saves.
//
// @param {import('@playwright/test').Page} page
// @param {{ title, tagline, description, price, duration, category?, difficulty?, thumbnailPath? }} courseDetails
// ---------------------------------------------------------------------------
async function createCourse(page, courseDetails) {
  // Navigate to the courses listing page
  const coursesPage = new CoursesPage(page);
  await coursesPage.navigate();

  // Click "New Course" to open the course creation form
  await coursesPage.clickNewCourse();

  // Fill all course details using the CourseFormPage object
  const courseForm = new CourseFormPage(page);
  await courseForm.fillCourseDetails(courseDetails);

  // Upload thumbnail if a file path was provided
  if (courseDetails.thumbnailPath) {
    await courseForm.uploadThumbnail(courseDetails.thumbnailPath);
  }

  // Save the course and navigate to the curriculum builder
  await courseForm.saveAndBuildCurriculum();
}

// ---------------------------------------------------------------------------
// addModule(page, moduleName) — Requirement 11.3
// Clicks "Add Module" on the curriculum page, enters the module name,
// and confirms the creation.
//
// @param {import('@playwright/test').Page} page
// @param {string} moduleName
// ---------------------------------------------------------------------------
async function addModule(page, moduleName) {
  // Delegate to the CurriculumPage page object
  const curriculumPage = new CurriculumPage(page);
  await curriculumPage.addModule(moduleName);
}

// ---------------------------------------------------------------------------
// addLesson(page, moduleName, lessonName, lessonType) — Requirement 11.4
// Clicks "Add Lesson" under the specified module, fills the lesson name,
// selects the lesson type, and confirms creation.
//
// @param {import('@playwright/test').Page} page
// @param {string} moduleName — the module to add the lesson under
// @param {string} lessonName — the name for the new lesson
// @param {'Video'|'Reading'|'Quiz'|'Coding Lab'} lessonType
// ---------------------------------------------------------------------------
async function addLesson(page, moduleName, lessonName, lessonType) {
  // Delegate to the CurriculumPage page object
  const curriculumPage = new CurriculumPage(page);
  await curriculumPage.addLesson(moduleName, lessonName, lessonType);
}

// Export all helper functions as named exports
module.exports = { login, createCourse, addModule, addLesson };
