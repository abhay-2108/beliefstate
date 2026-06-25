// Mobile menu toggle logic for beliefstate documentation
document.addEventListener('DOMContentLoaded', () => {
  const mobileToggle = document.getElementById('mobileMenuToggle');
  const mobileOverlay = document.getElementById('mobileMenuOverlay');
  
  if (mobileToggle && mobileOverlay) {
    // Toggle menu visibility
    mobileToggle.addEventListener('click', () => {
      const isActive = mobileToggle.classList.toggle('active');
      mobileOverlay.classList.toggle('active');
      document.body.classList.toggle('no-scroll', isActive);
    });

    // Close mobile menu when clicking any link
    mobileOverlay.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        mobileToggle.classList.remove('active');
        mobileOverlay.classList.remove('active');
        document.body.classList.remove('no-scroll');
      });
    });

    // Support ESC key to close the mobile menu
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && mobileOverlay.classList.contains('active')) {
        mobileToggle.classList.remove('active');
        mobileOverlay.classList.remove('active');
        document.body.classList.remove('no-scroll');
      }
    });
  }

  // Close mobile menu when resizing past the tablet breakpoint (1024px)
  window.addEventListener('resize', () => {
    if (window.innerWidth > 1024) {
      if (mobileToggle) mobileToggle.classList.remove('active');
      if (mobileOverlay) mobileOverlay.classList.remove('active');
      document.body.classList.remove('no-scroll');
    }
  });
});
