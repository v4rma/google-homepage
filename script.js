// Minimal interactivity for the Google homepage replica.

// "I'm Feeling Lucky" mimics Google's behavior: jump straight to the first
// result by using the search query with the btnI parameter.
function feelingLucky() {
  const input = document.querySelector('.search__input');
  const query = input ? input.value.trim() : '';
  const url = query
    ? `https://www.google.com/search?q=${encodeURIComponent(query)}&btnI=I`
    : 'https://www.google.com/doodles';
  window.location.href = url;
}
