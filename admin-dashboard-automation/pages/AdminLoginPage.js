// AdminLoginPage.js
// Page Object Model for the Owlet Campus admin login page at /admin/login.
// Locators use semantic Playwright APIs (getByLabel, getByPlaceholder, getByRole)
// — no CSS selectors — so they stay resilient to styling changes.
// Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 1.4, 1.6

class AdminLoginPage {
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

  /** Email field — uses placeholder text from the admin login form. */
  get emailInput() {
    // Try common admin login placeholders; the .or() chain handles variations
    return this.page
      .getByPlaceholder(/email/i)
      .or(this.page.getByPlaceholder('Email address'))
      .first();
  }

  /** Password field — uses placeholder text from the admin login form. */
  get passwordInput() {
    return this.page
      .getByPlaceholder(/password/i)
      .first();
  }

  /** Submit button — matches "Login", "Sign In", or "Log In" (case-insensitive). */
  get loginButton() {
    return this.page.getByRole('button', { name: /log in|login|sign in/i });
  }

  // ---------------------------------------------------------------------------
  // navigate() — Requirement 2.1
  // Navigates the browser to the admin login page and waits for the login
  // form to fully render (the page is a React SPA that shows "Loading..."
  // briefly before mounting the form).
  // ---------------------------------------------------------------------------
  async navigate() {
    await this.page.goto('/admin/login');
    // Wait for the email input to be visible — confirms the SPA has rendered
    await this.emailInput.waitFor({ state: 'visible' });
  }

  // ---------------------------------------------------------------------------
  // fillForm(email, password) — Requirement 2.2
  // Fills the email and password fields with the provided values.
  // ---------------------------------------------------------------------------
  async fillForm(email, password) {
    // Fill email field
    await this.emailInput.fill(email);
    // Fill password field
    await this.passwordInput.fill(password);
  }

  // ---------------------------------------------------------------------------
  // submit() — Requirement 2.3
  // Clicks the login/sign-in button to submit the form.
  // ---------------------------------------------------------------------------
  async submit() {
    await this.loginButton.click();
  }

  // ---------------------------------------------------------------------------
  // login(email, password) — Requirement 2.4
  // Convenience wrapper: fills the form then submits it in one call.
  // ---------------------------------------------------------------------------
  async login(email, password) {
    await this.fillForm(email, password);
    await this.submit();
  }
}

module.exports = { AdminLoginPage };
