// CurriculumPage.js
// Page Object Model for the Owlet Campus curriculum builder at /admin/courses/{id}/edit.
// Encapsulates module and lesson management interactions within the curriculum sidebar.
// Locators use semantic Playwright APIs — no CSS selectors — for resilience.
// Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 1.4, 1.6

class CurriculumPage {
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
   * Curriculum map container — Requirement 6.5
   * Targets the main curriculum builder container. Tries a heading/landmark
   * approach first, then falls back to common structural selectors.
   */
  get curriculumMap() {
    // Try a region/main landmark; fall back to common data-testid or class names
    return this.page
      .getByRole('main')
      .or(this.page.locator('[data-testid="curriculum-map"], .curriculum-map, .curriculum'))
      .first();
  }

  // ---------------------------------------------------------------------------
  // addModule(moduleName) — Requirement 6.1
  // Clicks the "Add Module" button, enters the module name in the input that
  // appears, and confirms the action by clicking the confirm/add button.
  // ---------------------------------------------------------------------------
  async addModule(moduleName) {
    // Click the "Add Module" button to open the module name input
    await this.page.getByRole('button', { name: /add module/i }).click();

    // Fill the module name in the input field that appears
    // Tries a dialog input first, then any visible text input
    const moduleInput = this.page
      .getByRole('dialog')
      .getByRole('textbox')
      .or(this.page.getByPlaceholder(/module name|enter module/i))
      .first();
    await moduleInput.fill(moduleName);

    // Confirm the module creation by clicking the confirm/add/save button
    await this.page
      .getByRole('button', { name: /confirm|add|save|create/i })
      .last()
      .click();
  }

  // ---------------------------------------------------------------------------
  // addLesson(moduleName, lessonName, lessonType) — Requirement 6.2
  // Clicks "Add Lesson" under the specified module, fills the lesson name,
  // selects the lesson type, and clicks "Create & Configure".
  // @param {string} moduleName — the module to add the lesson under
  // @param {string} lessonName — the name for the new lesson
  // @param {'Video'|'Reading'|'Quiz'|'Coding Lab'} lessonType — lesson type
  // ---------------------------------------------------------------------------
  async addLesson(moduleName, lessonName, lessonType) {
    // Find the module entry in the sidebar and click its "Add Lesson" button
    await this.getModuleLocator(moduleName)
      .locator('..')
      .getByRole('button', { name: /add lesson/i })
      .click();

    // Fill the lesson name in the input that appears
    const lessonNameInput = this.page
      .getByRole('dialog')
      .getByRole('textbox')
      .or(this.page.getByPlaceholder(/lesson name|enter lesson/i))
      .first();
    await lessonNameInput.fill(lessonName);

    // Select the lesson type — tries a radio button first, then a combobox
    const lessonTypeOption = this.page.getByRole('radio', { name: lessonType });
    const isRadio = await lessonTypeOption.count() > 0;
    if (isRadio) {
      // Radio button group for lesson type selection
      await lessonTypeOption.click();
    } else {
      // Combobox / select dropdown for lesson type
      const typeSelect = this.page
        .getByLabel(/lesson type|type/i)
        .or(this.page.getByRole('combobox', { name: /type/i }))
        .first();
      const tagName = await typeSelect.evaluate((el) => el.tagName.toLowerCase());
      if (tagName === 'select') {
        await typeSelect.selectOption(lessonType);
      } else {
        await typeSelect.click();
        await this.page.getByRole('option', { name: lessonType }).click();
      }
    }

    // Click "Create & Configure" to confirm lesson creation
    await this.page
      .getByRole('button', { name: /create.*configure|create & configure/i })
      .click();
  }

  // ---------------------------------------------------------------------------
  // getModuleLocator(moduleName) — Requirement 6.3
  // Returns a Playwright Locator for the module entry in the curriculum sidebar.
  // The caller can chain further assertions or interactions on the returned locator.
  // @param {string} moduleName
  // @returns {import('@playwright/test').Locator}
  // ---------------------------------------------------------------------------
  getModuleLocator(moduleName) {
    // Locate the module by its visible text in the sidebar
    return this.page.getByText(moduleName, { exact: false });
  }

  // ---------------------------------------------------------------------------
  // getLessonLocator(lessonName) — Requirement 6.4
  // Returns a Playwright Locator for the lesson entry in the curriculum sidebar.
  // @param {string} lessonName
  // @returns {import('@playwright/test').Locator}
  // ---------------------------------------------------------------------------
  getLessonLocator(lessonName) {
    // Locate the lesson by its visible text in the sidebar
    return this.page.getByText(lessonName, { exact: false });
  }
}

module.exports = { CurriculumPage };
