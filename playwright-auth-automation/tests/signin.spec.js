// signin.spec.js
// Sign-In test scenarios for the Owlet Campus login page.
// Requirements 5.1–5.5, 1.2, 1.5, 6.1, 6.3

const { test, expect } = require('@playwright/test');
const { SignInPage } = require('../pages/SignInPage');

// ---------------------------------------------------------------------------
// Test data constants
// ---------------------------------------------------------------------------

// Real registered account credentials
const TEST_EMAIL = process.env.TEST_EMAIL || 'hulk102004@gmail.com';
const TEST_PASSWORD = process.env.TEST_PASSWORD || 'Bhargav@2004';

// Non-existent account (Req 5.2)
const INVALID_EMAIL = 'nonexistent_xyz_99@example.com';
const INVALID_PASSWORD = 'WrongPass@000';

// Real email + wrong password (Req 5.4)
const REGISTERED_EMAIL = 'hulk102004@gmail.com';
const WRONG_PASSWORD = 'IncorrectPass@999';

// ---------------------------------------------------------------------------
// Shared page object reference
// ---------------------------------------------------------------------------
let signInPage;

// beforeEach hook — Requirement 5.5
test.beforeEach(async ({ page }) => {
  signInPage = new SignInPage(page);
  await signInPage.navigate();
});

// ---------------------------------------------------------------------------
// Helper: wait for any visible alert/toast/error on the page
// ---------------------------------------------------------------------------
async function getAnyAlert(page) {
  return page.locator(
    '[role="alert"], .toast, .notification, .error, .alert, ' +
    '[class*="toast"], [class*="error"], [class*="alert"], [class*="notification"]'
  ).first();
}

// ---------------------------------------------------------------------------
// Positive scenario — Requirement 5.1
// ---------------------------------------------------------------------------
test('should login successfully with valid credentials', async ({ page }) => {
  await signInPage.login(TEST_EMAIL, TEST_PASSWORD);

  // After successful login, URL should change away from /user/login
  await expect(page).not.toHaveURL(/\/user\/login/, { timeout: 15000 });
});

// ---------------------------------------------------------------------------
// Negative scenarios
// ---------------------------------------------------------------------------

// Requirement 5.2
test('should show error for invalid credentials', async ({ page }) => {
  await signInPage.login(INVALID_EMAIL, INVALID_PASSWORD);

  // Any alert/toast should appear indicating failure
  const alert = await getAnyAlert(page);
  await expect(alert).toBeVisible({ timeout: 10000 });
});

// Requirement 5.3
test('should show validation message for empty form', async ({ page }) => {
  // Click Sign In without filling any fields
  await signInPage.submit();

  // Browser native validation or app-level alert should trigger
  const alert = await getAnyAlert(page);
  const inlineError = page.locator('[class*="error"], [class*="invalid"], input:invalid').first();

  const alertVisible = await alert.isVisible({ timeout: 8000 }).catch(() => false);
  const inlineVisible = await inlineError.isVisible().catch(() => false);

  expect(alertVisible || inlineVisible).toBe(true);
});

// Requirement 5.4
test('should show error for incorrect password', async ({ page }) => {
  await signInPage.login(REGISTERED_EMAIL, WRONG_PASSWORD);

  // Any alert/toast should appear indicating authentication failure
  const alert = await getAnyAlert(page);
  await expect(alert).toBeVisible({ timeout: 10000 });
});
