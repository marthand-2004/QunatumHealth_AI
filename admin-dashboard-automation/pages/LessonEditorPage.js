// LessonEditorPage.js
// Page Object Model for the Owlet Campus lesson content editor.
// Encapsulates configuration interactions for all four lesson types:
// Video, Reading, Quiz, and Coding Lab.
// Locators use semantic Playwright APIs — no CSS selectors — for resilience.
// Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 1.4, 1.6

class LessonEditorPage {
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
   * Video URL input — Requirement 7.1
   * Matches a label or placeholder containing "video url" or "url".
   */
  get videoUrlInput() {
    return this.page
      .getByLabel(/video url|url/i)
      .or(this.page.getByPlaceholder(/video url|enter url|paste url/i))
      .first();
  }

  /**
   * Reading content editor — Requirement 7.2
   * Targets a rich text editor (contenteditable div) or a textarea.
   * Falls back to a role="textbox" which covers most rich text editors.
   */
  get readingEditor() {
    return this.page
      .getByRole('textbox', { name: /content|editor|reading/i })
      .or(this.page.locator('[contenteditable="true"]'))
      .first();
  }

  /**
   * Quiz question input — Requirement 7.3
   * Matches a label or placeholder containing "question".
   */
  get quizQuestionInput() {
    return this.page
      .getByLabel(/question/i)
      .or(this.page.getByPlaceholder(/question|enter question/i))
      .first();
  }

  /**
   * Code editor element — Requirement 7.4
   * Targets Monaco or CodeMirror editors which render as contenteditable
   * regions or textareas inside their container.
   */
  get codeEditor() {
    return this.page
      .locator('.monaco-editor textarea, .CodeMirror textarea, [data-testid="code-editor"] textarea')
      .or(this.page.locator('[contenteditable="true"].view-line').locator('..'))
      .first();
  }

  // ---------------------------------------------------------------------------
  // configureVideoLesson(videoUrl) — Requirement 7.1
  // Enters the video URL in the URL input field and saves the lesson.
  // @param {string} videoUrl — the video URL to enter
  // ---------------------------------------------------------------------------
  async configureVideoLesson(videoUrl) {
    // Enter the video URL
    await this.videoUrlInput.fill(videoUrl);

    // Click verify/save button to confirm the URL
    await this.page
      .getByRole('button', { name: /verify|save|confirm/i })
      .first()
      .click();
  }

  // ---------------------------------------------------------------------------
  // configureReadingLesson(content) — Requirement 7.2
  // Enters text content in the reading editor and saves the lesson.
  // @param {string} content — the reading content to enter
  // ---------------------------------------------------------------------------
  async configureReadingLesson(content) {
    // Click the editor to focus it, then fill with content
    await this.readingEditor.click();
    await this.readingEditor.fill(content);

    // Save the reading lesson content
    await this.page
      .getByRole('button', { name: /save/i })
      .first()
      .click();
  }

  // ---------------------------------------------------------------------------
  // configureQuizLesson(question, options, correctIndex) — Requirement 7.3
  // Adds a question, adds all answer options, marks the correct answer,
  // and saves/publishes the quiz.
  // @param {string} question — the quiz question text
  // @param {string[]} options — array of answer option strings
  // @param {number} correctIndex — zero-based index of the correct option
  // ---------------------------------------------------------------------------
  async configureQuizLesson(question, options, correctIndex) {
    // Enter the quiz question text
    await this.quizQuestionInput.fill(question);

    // Add each answer option
    for (let i = 0; i < options.length; i++) {
      // Click "Add Option" button to create a new option input
      await this.page
        .getByRole('button', { name: /add option|add answer/i })
        .click();

      // Fill the option text in the most recently added option input
      const optionInputs = this.page.getByPlaceholder(/option|answer/i);
      await optionInputs.nth(i).fill(options[i]);
    }

    // Mark the correct answer by clicking the radio/checkbox at correctIndex
    const correctMarkers = this.page
      .getByRole('radio', { name: /correct/i })
      .or(this.page.getByRole('checkbox', { name: /correct/i }));
    await correctMarkers.nth(correctIndex).click();

    // Save/publish the quiz
    await this.page
      .getByRole('button', { name: /publish|save/i })
      .first()
      .click();
  }

  // ---------------------------------------------------------------------------
  // configureCodingLabLesson(starterCode, instructions) — Requirement 7.4
  // Enters starter code in the code editor and instructions, then saves.
  // @param {string} starterCode — the starter code to enter in the editor
  // @param {string} instructions — the instructions text
  // ---------------------------------------------------------------------------
  async configureCodingLabLesson(starterCode, instructions) {
    // Focus and fill the code editor with starter code
    // Monaco/CodeMirror editors require clicking first to focus
    await this.codeEditor.click();
    // Select all existing content and replace with starter code
    await this.page.keyboard.press('Control+A');
    await this.page.keyboard.type(starterCode);

    // Fill the instructions field if present
    const instructionsInput = this.page
      .getByLabel(/instructions/i)
      .or(this.page.getByPlaceholder(/instructions/i))
      .first();
    const instructionsCount = await instructionsInput.count();
    if (instructionsCount > 0) {
      await instructionsInput.fill(instructions);
    }

    // Save the coding lab lesson
    await this.page
      .getByRole('button', { name: /save/i })
      .first()
      .click();
  }
}

module.exports = { LessonEditorPage };
