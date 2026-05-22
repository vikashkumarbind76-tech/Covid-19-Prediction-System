# TODO - Premium COVID-19 Prediction Dashboard

- [x] Rewrite `index.html` to match the requested premium Awwwards-style UI
  - [ ] Update theme: Outfit font, FontAwesome, Chart.js, noise overlay, floating blobs, emerald accents
  - [ ] Implement fixed header + scroll shrink + active nav underline animation
  - [ ] Implement mobile full-screen overlay menu
  - [ ] Build hero + intersection observer animated counters
  - [ ] Build Prediction Engine with left inputs + right output (Ready → Loading 2.5s → Result)
  - [ ] Implement custom Gender dropdown with click-away close and synchronized trigger
  - [ ] Implement symptom selectors (Fever, Cough, Fatigue, Chest Pain, Breathing Difficulty, Diabetes) with Yes/No dynamic styling (Yes=green glow, No=red glow)
  - [ ] Implement risk simulation logic based on active symptom count (>=4 HIGH, 2-3 MEDIUM, <2 LOW) with confidence ranges
  - [ ] Implement confidence progress bar that matches risk confidence
  - [ ] Implement Prediction History using localStorage key `covidHistory` (load, render, clear with confirmation)
  - [ ] Implement Chart.js line chart for Global Recovery Rate with dark tooltip styling
  - [ ] Implement Safety Protocol 3-column cards with FontAwesome icons
  - [ ] Implement footer with educational-only medical disclaimer
- [ ] Smoke test in browser (manual): responsiveness, animations, dropdown behavior, localStorage persistence, and chart rendering

