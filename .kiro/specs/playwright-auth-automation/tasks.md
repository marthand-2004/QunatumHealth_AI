# Implementation Plan: Playwright Auth Automation

## Overview

Implement a standalone Playwright + JavaScript test suite for the Owlet Campus Sign-Up and Sign-In modules using the Page Object Model pattern. Tasks progress from project scaffolding through page objects, example-based tests, and property-based tests.

## Tasks

- [x] 1. Scaffold project structure and configuration
  - Create `playwright-auth-automation/` directory with `package.json` declaring `@playwright/test` and `fast-check` as dev dependencies
  - Create `playwright.config.js` setting `baseURL`, `testDir`, headless mode, and HTML reporter (`open: 'never'`)
  - Create empty `pages/` and `tests/` directories
  - _Requirements: 1.1, 1.2, 1.3, 6.4_

- [x] 2. Implement SignUpPage page object
  - [x] 2.1 Create `pages/SignUpPage.js` with constructor, locator getters, `navigate()`, `fillForm()`, `submit()`, and `register()` methods
    - Use `page.locator()` or `getByRole()` for all element selection
    - Add inline comments explaining each logical block
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 1.4, 1.5_

  - [ ]* 2.2 Write property test for SignUpPage fillForm round-trip
    - **Property 1: SignUpPage fillForm round-trip**
    - **Validates: Requirements 2.2**

- [x] 3. Implement SignInPage page object
  - [x] 3.1 Create `pages/SignInPage.js` with constructor, locator getters, `navigate()`, `fillForm()`, `submit()`, and `login()` methods
    - Use `page.locator()` or `getByRole()` for all element selection
    - Add inline comments explaining each logical block
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 1.4, 1.5_

  - [ ]* 3.2 Write property test for SignInPage fillForm round-trip
    - **Property 2: SignInPage fillForm round-trip**
    - **Validates: Requirements 3.2**

- [x] 4. Checkpoint — Ensure page objects are wired and locator tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Sign-Up example-based tests
  - [x] 5.1 Create `tests/signup.spec.js` with `test.beforeEach` hook that instantiates `SignUpPage` and calls `navigate()`
    - _Requirements: 4.6_

  - [x] 5.2 Write `should register successfully with valid data` test
    - Assert navigation away from registration page or visible success indicator after submitting valid data
    - _Requirements: 4.1_

  - [x] 5.3 Write `should show error for duplicate email` test
    - Assert duplicate-account error message is visible
    - _Requirements: 4.2_

- [ ] 6. Implement Sign-Up property-based tests
  - [ ]* 6.1 Write property test for invalid email format
    - **Property 3: Invalid email format always triggers validation error**
    - **Validates: Requirements 4.3**

  - [ ]* 6.2 Write property test for mismatched passwords
    - **Property 4: Mismatched passwords always trigger a mismatch error**
    - **Validates: Requirements 4.4**

  - [ ]* 6.3 Write property test for empty required fields
    - **Property 5: Empty required fields always trigger validation messages**
    - **Validates: Requirements 4.5**

- [x] 7. Implement Sign-In example-based tests
  - [x] 7.1 Create `tests/signin.spec.js` with `test.beforeEach` hook that instantiates `SignInPage` and calls `navigate()`
    - _Requirements: 5.5_

  - [x] 7.2 Write `should login successfully with valid credentials` test
    - Read credentials from `process.env.TEST_EMAIL` / `process.env.TEST_PASSWORD`; skip with `test.skip()` if undefined
    - Assert navigation to authenticated home or dashboard page
    - _Requirements: 5.1_

  - [x] 7.3 Write `should show validation message for empty form` test
    - Assert required-field or validation message is visible when both fields are empty
    - _Requirements: 5.3_

- [ ] 8. Implement Sign-In property-based tests
  - [ ]* 8.1 Write property test for invalid credentials
    - **Property 6: Invalid credentials always produce an authentication error**
    - **Validates: Requirements 5.2, 5.4**

- [x] 9. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use `fast-check` with a minimum of 100 iterations (`{ numRuns: 100 }`)
- Each property test file includes a comment: `// Feature: playwright-auth-automation, Property N: <title>`
- CI credentials must be supplied via `TEST_EMAIL` and `TEST_PASSWORD` environment variables
