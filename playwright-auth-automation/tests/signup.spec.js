// signup.spec.js
// Sign-Up test scenarios for the Owlet Campus registration page.
// Requirements 4.1–4.6, 1.2, 1.5, 6.1, 6.3

const { test, expect } = require('@playwright/test');
const { SignUpPage } = require('../pages/SignUpPage');

// ---------------------------------------------------------------------------
// Test data constants
// ---------------------------------------------------------------------------

// Unique email + mobile per run to avoid duplicate conflicts (Req 4.1)
const timestamp = Date.now();
const VALID_USER = {
  name: 'Test User',
  email: `testuser+${timestamp}@example.com`,
  mobile: `9${String(timestamp).slice(-9)}`, // 10-digit mobile
  password: 'ValidPass@123',
  confirmPassword: 'ValidPass@123',
};

// Known duplicate — already registered account (Req 4.2)
const DUPLICATE_EMAIL = 'hulk102004@gmail.com';

// Malformed email (Req 4.3)
const INVALID_EMAIL = 'not-an-email';

// Mismatched passwords (Req 4.4)
const MISMATCH = {
  password: 'Pass123!',
  confirmPassword: 'Different456!',
};

// ---------------------------------------------------------------------------
// Shared page object reference
// ---------------------------------------------------------------------------
let signUpPage;

// beforeEach hook — Requirement 4.6
test.beforeEach(async ({ page }) => {
  signUpPage = new SignUpPage(page);
  await signUpPage.navigate();
});

// ---------------------------------------------------------------------------
// Helper: wait for any visible alert/toast/error on the page
// Covers toast libraries, inline errors, and aria alerts
// ---------------------------------------------------------------------------
async function getAnyAlert(page) {
  return page.locator(
    '[role="alert"], .toast, .notification, .error, .alert, ' +
    '[class*="toast"], [class*="error"], [class*="alert"], [class*="notification"]'
  ).first();
}

// ---------------------------------------------------------------------------
// Positive scenario — Requirement 4.1
// ---------------------------------------------------------------------------
test('should register successfully with valid data', async ({ page }) => {
  await signUpPage.register(
    VALID_USER.name,
    VALID_USER.email,
    VALID_USER.password,
    VALID_USER.confirmPassword,
    VALID_USER.mobile
  );

  // Success = navigated away from signup page OR a success alert is visible
  const alert = await getAnyAlert(page);
  const alertVisible = await alert.isVisible().catch(() => false);

  if (!alertVisible) {
    await expect(page).not.toHaveURL(/\/user\/signup/, { timeout: 10000 });
  } else {
    // If still on signup page, the alert should NOT contain an error keyword
    const alertText = await alert.innerText().catch(() => '');
    expect(alertText.toLowerCase()).not.toMatch(/error|fail|invalid|wrong/);
  }
});

// ---------------------------------------------------------------------------
// Negative scenarios
// ---------------------------------------------------------------------------

// Requirement 4.2
test('should show error for duplicate email', async ({ page }) => {
  await signUpPage.register(
    'Existing User',
    DUPLICATE_EMAIL,
    'SomePass@123',
    'SomePass@123',
    '9876543210'
  );

  // Wait for any alert/toast to appear after submission
  const alert = await getAnyAlert(page);
  await expect(alert).toBeVisible({ timeout: 10000 });
});

// Requirement 4.3
test('should show error for invalid email format', async ({ page }) => {
  await signUpPage.register(
    'Test User',
    INVALID_EMAIL,
    'SomePass@123',
    'SomePass@123',
    '9876543210'
  );

  // Either an inline validation message or an alert should appear
  const alert = await getAnyAlert(page);
  const inlineError = page.locator('input:invalid, [class*="error"], [class*="invalid"]').first();

  const alertVisible = await alert.isVisible({ timeout: 8000 }).catch(() => false);
  const inlineVisible = await inlineError.isVisible().catch(() => false);

  expect(alertVisible || inlineVisible).toBe(true);
});

// Requirement 4.4
test('should show error for mismatched passwords', async ({ page }) => {
  await signUpPage.register(
    'Test User',
    `mismatch+${timestamp}@example.com`,
    MISMATCH.password,
    MISMATCH.confirmPassword,
    '9876543211'
  );

  const alert = await getAnyAlert(page);
  const inlineError = page.locator('[class*="error"], [class*="invalid"]').first();

  const alertVisible = await alert.isVisible({ timeout: 8000 }).catch(() => false);
  const inlineVisible = await inlineError.isVisible().catch(() => false);

  expect(alertVisible || inlineVisible).toBe(true);
});

// Requirement 4.5
test('should show required field errors for empty form submission', async ({ page }) => {
  // Click submit without filling anything
  await signUpPage.submit();

  // Browser native validation or app-level validation should trigger
  const alert = await getAnyAlert(page);
  const inlineError = page.locator('[class*="error"], [class*="invalid"], input:invalid').first();

  const alertVisible = await alert.isVisible({ timeout: 8000 }).catch(() => false);
  const inlineVisible = await inlineError.isVisible().catch(() => false);

  expect(alertVisible || inlineVisible).toBe(true);
});
