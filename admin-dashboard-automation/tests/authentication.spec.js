// authentication.spec.js
// Test suite for the Owlet Campus admin authentication flow.
// Covers: successful login, sidebar visibility, user identity display,
// and invalid credentials error handling.
// Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 12.1, 12.3, 12.5, 1.7

const { test, expect } = require('@playwright/test');
const { AdminLoginPage } = require('../pages/AdminLoginPage');
const { DashboardPage } = require('../pages/DashboardPage');

// ---------------------------------------------------------------------------
// Admin credentials — read from environment variables for CI safety.
// Falls back to hardcoded constants for local development runs.
// ---------------------------------------------------------------------------
const ADMIN_EMAIL    = process.env.ADMIN_EMAIL    || 'raghuram@gmail.com';
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'Ruaf@1489';

test.describe('Authentication', () => {
  // -------------------------------------------------------------------------
  // beforeEach — Requirement 8.5
  // Navigate to the admin login page before every test in this describe block.
  // -------------------------------------------------------------------------
  test.beforeEach(async ({ page }) => {
    const loginPage = new AdminLoginPage(page);
    await loginPage.navigate();
  });

  // -------------------------------------------------------------------------
  // Test 1: Admin login page loads correctly
  // Verifies the login form is visible after navigating to /admin/login.
  // -------------------------------------------------------------------------
  test('should navigate to admin login page', async ({ page }) => {
    const loginPage = new AdminLoginPage(page);

    // Assert the email and password fields are visible — navigate() already waited
    await expect(loginPage.emailInput).toBeVisible({ timeout: 15000 });
    await expect(loginPage.passwordInput).toBeVisible({ timeout: 15000 });
    await expect(loginPage.loginButton).toBeVisible({ timeout: 15000 });
  });

  // -------------------------------------------------------------------------
  // Test 2: Successful login redirects to dashboard — Requirement 8.1
  // Submits valid admin credentials and asserts the URL changes to the dashboard.
  // -------------------------------------------------------------------------
  test('should login successfully with valid credentials', async ({ page }) => {
    const loginPage = new AdminLoginPage(page);

    // Submit valid admin credentials
    await loginPage.login(ADMIN_EMAIL, ADMIN_PASSWORD);

    // Assert the URL has changed away from the login page (dashboard redirect)
    await expect(page).not.toHaveURL(/\/admin\/login/);

    // Assert the URL contains the admin area (dashboard or any admin route)
    await expect(page).toHaveURL(/\/admin/);
  });

  // -------------------------------------------------------------------------
  // Test 3: Sidebar navigation is visible after login — Requirement 8.2
  // Confirms the sidebar renders after a successful authentication.
  // -------------------------------------------------------------------------
  test('should show sidebar navigation after login', async ({ page }) => {
    const loginPage = new AdminLoginPage(page);
    const dashboard = new DashboardPage(page);

    // Log in with valid credentials
    await loginPage.login(ADMIN_EMAIL, ADMIN_PASSWORD);

    // Wait for and assert the sidebar navigation is visible
    await dashboard.waitForLoad();
    await expect(dashboard.sidebarNav).toBeVisible();
  });

  // -------------------------------------------------------------------------
  // Test 4: User identity display is visible after login — Requirement 8.3
  // Confirms the user name / avatar is rendered in the top-right area.
  // -------------------------------------------------------------------------
  test('should show user identity display after login', async ({ page }) => {
    const loginPage = new AdminLoginPage(page);
    const dashboard = new DashboardPage(page);

    // Log in with valid credentials
    await loginPage.login(ADMIN_EMAIL, ADMIN_PASSWORD);

    // Wait for the dashboard to load
    await dashboard.waitForLoad();

    // Assert the user identity element is visible in the top-right area
    await expect(dashboard.userIdentity).toBeVisible();
  });

  // -------------------------------------------------------------------------
  // Test 5: Invalid credentials show an error message — Requirement 8.4
  // Submits a wrong email/password and asserts an error message appears.
  // -------------------------------------------------------------------------
  test('should show error message for invalid credentials', async ({ page }) => {
    const loginPage = new AdminLoginPage(page);

    // Submit credentials that are guaranteed to be invalid
    await loginPage.login('invalid@example.com', 'WrongPassword123!');

    // Assert an error message is visible on the page
    // Matches common error patterns: "invalid", "incorrect", "wrong", "failed"
    const errorMessage = page
      .getByRole('alert')
      .or(page.getByText(/invalid|incorrect|wrong|failed|error/i))
      .first();
    await expect(errorMessage).toBeVisible();

    // Assert we are still on the login page (no redirect occurred)
    await expect(page).toHaveURL(/\/admin\/login/);
  });
});
