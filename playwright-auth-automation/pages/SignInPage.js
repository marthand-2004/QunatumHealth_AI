// SignInPage.js
// Page Object Model for the Owlet Campus login page.
// Placeholders confirmed from UI: 'Email address', 'Password'. Button: 'Sign In'
// Requirements 3.1–3.6, 1.4, 1.5

class SignInPage {
  /** @param {import('@playwright/test').Page} page */
  constructor(page) {
    this.page = page;
  }

  // ---------------------------------------------------------------------------
  // Locator accessors — using exact placeholder text from the live UI
  // ---------------------------------------------------------------------------

  get emailInput() {
    return this.page.getByPlaceholder('Email address');
  }

  get passwordInput() {
    return this.page.getByPlaceholder('Password');
  }

  get loginButton() {
    // Button text is 'Sign In' on this app
    return this.page.getByRole('button', { name: /sign in/i });
  }

  // ---------------------------------------------------------------------------
  // Navigation — Requirement 3.1
  // ---------------------------------------------------------------------------
  async navigate() {
    await this.page.goto('/user/login');
    // Wait for the Email address field to confirm the form has rendered
    await this.page.getByPlaceholder('Email address').waitFor({ timeout: 15000 });
  }

  // ---------------------------------------------------------------------------
  // Form interactions — Requirement 3.2
  // ---------------------------------------------------------------------------
  async fillForm(email, password) {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
  }

  // ---------------------------------------------------------------------------
  // Submission — Requirement 3.3
  // ---------------------------------------------------------------------------
  async submit() {
    await this.loginButton.click();
  }

  // ---------------------------------------------------------------------------
  // Convenience method — Requirement 3.4
  // ---------------------------------------------------------------------------
  async login(email, password) {
    await this.fillForm(email, password);
    await this.submit();
  }
}

module.exports = { SignInPage };
