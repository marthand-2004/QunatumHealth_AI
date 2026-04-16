// CoursesPage.js
// Page Object Model for the Owlet Campus admin courses listing at /admin/courses.
// Encapsulates the course list, the "New Course" entry point, and per-course
// search and edit interactions.
// Locators use semantic Playwright APIs — no CSS selectors — for resilience.
// Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 1.4, 1.6

class CoursesPage {
  /** @param {import('@playwright/test').Page} page */
  constructor(page) {
    this.page = page;
  }

  // ---------------------------------------------------------------------------
  // Locator accessors
  // Getters return a fresh Playwright Locator each time they are called.
  // Playwright locator errors (e.g. TimeoutError) are intentionally not caught
  // here — they propagate to the caller so failures are visible in test output.
  // ---------------------------------------------------------------------------

  /**
   * Course list container — Requirement 4.1
   * Tries the ARIA list role first (semantic HTML), then falls back to common
   * data-testid / class / structural selectors used by the Owlet Campus UI.
   */
  get courseList() {
    // getByRole('list') matches a <ul> or <ol> landmark; the chained .or()
    // fallbacks cover custom containers that don't use a native list element
    return this.page
      .getByRole('list')
      .or(
        this.page.locator(
          '[data-testid="course-list"], .course-list, main ul, main'
        )
      )
      .first();
  }

  /**
   * "New Course" button or link — Requirement 4.2
   * Matches either a <button> or an <a> element whose accessible name contains
   * "new course" (case-insensitive), covering both button and link variants.
   */
  get newCourseButton() {
    // Try a button first; fall back to a link with the same label
    return this.page
      .getByRole('button', { name: /new course/i })
      .or(this.page.getByRole('link', { name: /new course/i }))
      .first();
  }

  // ---------------------------------------------------------------------------
  // navigate() — Requirement 4.3
  // Navigates the browser to the admin courses listing page.
  // ---------------------------------------------------------------------------
  async navigate() {
    await this.page.goto('/admin/courses');
  }

  // ---------------------------------------------------------------------------
  // clickNewCourse() — Requirement 4.4
  // Clicks the "New Course" button/link to open the course creation form.
  // ---------------------------------------------------------------------------
  async clickNewCourse() {
    await this.newCourseButton.click();
  }

  // ---------------------------------------------------------------------------
  // searchCourse(title) — Requirement 4.5
  // Returns a Playwright Locator for the course card whose visible text
  // contains the given title (case-sensitive, partial match).
  // The caller can chain further assertions or interactions on the returned
  // locator without triggering a network request.
  // ---------------------------------------------------------------------------
  searchCourse(title) {
    // getByText with exact: false performs a substring match, so partial titles
    // (e.g. a truncated display name) still resolve to the correct card
    return this.page.getByText(title, { exact: false });
  }

  // ---------------------------------------------------------------------------
  // clickEditCourse(title) — Requirement 4.5
  // Finds the course card matching the given title, then clicks the edit icon
  // or button rendered adjacent to that card.
  // Errors from Playwright locators propagate — they are not caught here.
  // ---------------------------------------------------------------------------
  async clickEditCourse(title) {
    // Locate the text node for the course title, traverse up to its parent
    // container (..), then find the edit button within that container
    await this.page
      .getByText(title)
      .locator('..')
      .getByRole('button', { name: /edit/i })
      .click();
  }
}

module.exports = { CoursesPage };
