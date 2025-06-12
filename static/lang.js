function setupTranslations(dict) {
    const switcher = document.getElementById('lang-switcher');
    const htmlEl = document.documentElement;

    function apply(lang) {
        document.querySelectorAll('[data-lang]').forEach(el => {
            const key = el.getAttribute('data-lang');
            if (dict[lang] && dict[lang][key]) {
                if (el.placeholder !== undefined) {
                    el.placeholder = dict[lang][key];
                } else {
                    el.innerHTML = dict[lang][key];
                }
            }
        });
        htmlEl.lang = lang;
        htmlEl.dir = lang === 'ar' ? 'rtl' : 'ltr';
        if (switcher) switcher.textContent = lang === 'ar' ? 'English' : 'العربية';
        localStorage.setItem('lang', lang);
    }

    if (switcher) {
        switcher.addEventListener('click', () => {
            const newLang = htmlEl.lang === 'ar' ? 'en' : 'ar';
            apply(newLang);
        });
    }

    document.addEventListener('DOMContentLoaded', () => {
        const saved = localStorage.getItem('lang') || htmlEl.lang || 'en';
        apply(saved);
    });
}
