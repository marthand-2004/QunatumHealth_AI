// SignUpPage.js
// Page Object Model for the Owlet Campus registration page.
// Placeholders confirmed from UI: 'Full Name', 'Email address', 'Mobile Number',
// 'Password', 'Confirm Password'. Button text: 'Sign Up'
// Requirements 2.1–2.6, 1.4, 1.5

class SignUpPage {
  /** @param {import('@playwright/test').Page} page */
  constructor(page) {
    this.page = page;
  }

  // ---------------------------------------------------------------------------
  // Locator accessors — using exact placeholder text from the live UI
  // ---------------------------------------------------------------------------

  get nameInput() {
    return this.page.getByPlaceholder('Full Name');
  }

  get emailInput() {
    return this.page.getByPlaceholder('Email address');
  }

  get mobileInput() {
    return this.page.getByPlaceholder('Mobile Number');
  }

  get passwordInput() {
    // First password field
    return this.page.locator('input[type="password"]').first();
  }

  get confirmInput() {
    // Second password field (Confirm Password)
    return this.page.locator('input[type="password"]').nth(1);
  }

  get registerButton() {
    // Button text is 'Sign Up' on this app
    return this.page.getByRole('button', { name: /sign up/i });
  }

  // ---------------------------------------------------------------------------
  // Navigation — Requirement 2.1
  // ---------------------------------------------------------------------------
  async navigate() {
    await this.page.goto('/user/signup');
    // Wait for the Full Name field to confirm the form has rendered
    await this.page.getByPlaceholder('Full Name').waitFor({ timeout: 15000 });
  }

  // ---------------------------------------------------------------------------
  // Form interactions — Requirement 2.2
  // Accepts optional mobile number (defaults to a placeholder value)
  // ---------------------------------------------------------------------------
  async fillForm(name, email, password, confirmPassword, mobile = '9999999999') {
    await this.nameInput.fill(name);
    await this.emailInput.fill(email);
    await this.mobileInput.fill(mobile);
    await this.passwordInput.fill(password);
    await this.confirmInput.fill(confirmPassword);
  }

  // ---------------------------------------------------------------------------
  // Submission — Requirement 2.3
  // ---------------------------------------------------------------------------
  async submit() {
    await this.registerButton.click();
  }

  // ---------------------------------------------------------------------------
  // Convenience method — Requirement 2.4
  // ---------------------------------------------------------------------------
  async register(name, email, password, confirmPassword, mobile = '9999999999') {
    await this.fillForm(name, email, password, confirmPassword, mobile);
    await this.submit();
  }
}

module.exports = { SignUpPage };
