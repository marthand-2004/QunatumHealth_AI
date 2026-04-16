// CourseFormPage.js
// Page Object Model for the Owlet Campus new/edit course form.
// Encapsulates all form field interactions, thumbnail upload, and the
// "Save and Build Curriculum" submission action.
// Locators use semantic Playwright APIs (getByLabel, getByPlaceholder,
// getByRole) — no CSS selectors — for resilience to styling changes.
// Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 1.4, 1.6

class CourseFormPage {
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
   * Course title input.
   * DOM: textbox with placeholder "Enter course title"
   */
  get titleInput() {
    return this.page.getByPlaceholder('Enter course title');
  }

  /**
   * Tagline input.
   * DOM: textbox with placeholder "Short description for card views..."
   */
  get taglineInput() {
    return this.page.getByPlaceholder(/short description for card views/i);
  }

  /**
   * Description input.
   * DOM: textbox with placeholder "Describe what students will learn…"
   * The description field is a rich-text textbox (role="textbox").
   */
  get descriptionInput() {
    return this.page.getByPlaceholder(/describe what students will learn/i);
  }

  /**
   * Price input.
   * DOM: spinbutton inside the "Price (USD)" section — no label/placeholder.
   * Targeted by role="spinbutton" scoped to the price container.
   */
  get priceInput() {
    return this.page.getByRole('spinbutton').first();
  }

  /**
   * Duration input.
   * DOM: spinbutton inside the "Duration (Hours)" section — second spinbutton.
   */
  get durationInput() {
    return this.page.getByRole('spinbutton').nth(1);
  }

  /**
   * Category combobox.
   * DOM: native <select> combobox with options like "General", "CAT Prep", etc.
   */
  get categorySelect() {
    return this.page.getByRole('combobox').first();
  }

  /**
   * Difficulty level — radio buttons (Beginner / Intermediate / Advanced).
   * DOM: radio group, not a select. Use selectDifficulty() method instead.
   * This getter returns the radio group container for reference.
   */
  get difficultySelect() {
    return this.page.getByRole('radio', { name: /beginner|intermediate|advanced/i }).first();
  }

  /**
   * Thumbnail file input — file inputs have no ARIA role, so a CSS attribute
   * selector is the only reliable approach here.
   */
  get thumbnailInput() {
    return this.page.locator('input[type="file"]');
  }

  // ---------------------------------------------------------------------------
  // fillCourseDetails(details) — Requirements 5.1, 5.2
  // Fills all text fields and selects category/difficulty dropdown values.
  // Handles both native <select> elements and custom dropdown widgets.
  // @param {{ title, tagline, description, price, duration, category?, difficulty? }} details
  // ---------------------------------------------------------------------------
  async fillCourseDetails(details) {
    // Fill the course title
    await this.titleInput.fill(details.title);

    // Fill the tagline (placeholder: "Short description for card views...")
    await this.taglineInput.fill(details.tagline);

    // Fill the description rich-text textbox
    await this.descriptionInput.fill(details.description);

    // Fill price — spinbutton, clear first then type
    await this.priceInput.fill(details.price);

    // Fill duration — second spinbutton
    await this.durationInput.fill(details.duration);

    // Category — native <select> combobox, use selectOption directly
    if (details.category) {
      await this.categorySelect.selectOption(details.category);
    }

    // Difficulty — radio buttons (Beginner / Intermediate / Advanced)
    // Click the radio whose label matches the requested difficulty
    if (details.difficulty) {
      await this.page.getByRole('radio', { name: details.difficulty }).click();
    }
  }

  // ---------------------------------------------------------------------------
  // uploadThumbnail(filePath) — Requirement 5.3
  // Sets the file input to the given file path. Works even when the input is
  // hidden behind a styled upload button.
  // @param {string} filePath — absolute or workspace-relative path to the image
  // ---------------------------------------------------------------------------
  async uploadThumbnail(filePath) {
    await this.thumbnailInput.setInputFiles(filePath);
  }

  // ---------------------------------------------------------------------------
  // saveAndBuildCurriculum() — Requirement 5.4
  // Clicks the "Save and Build Curriculum" button to submit the course form
  // and navigate to the curriculum builder.
  // ---------------------------------------------------------------------------
  async saveAndBuildCurriculum() {
    // Button text in DOM: "Save & Build Curriculum"
    await this.page
      .getByRole('button', { name: /save.*build curriculum/i })
      .click();
  }
}

module.exports = { CourseFormPage };
