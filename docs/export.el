;; Batch export org-mode files to RST for Sphinx
;; Usage: emacs --batch -l docs/export.el

;; Setup Package Manager (to fetch ox-rst automatically)
(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
(package-initialize)

;; Ensure ox-rst is present
(unless (package-installed-p 'ox-rst)
  (package-refresh-contents)
  (package-install 'ox-rst))

(require 'ox-rst)
(require 'ox-publish)

;; Define the Publishing Project
(setq org-publish-project-alist
      '(("sphinx-rst"
         :base-directory "./docs/orgmode/"
         :base-extension "org"
         :publishing-directory "./docs/source/"
         :publishing-function org-rst-publish-to-rst
         :recursive t
         :headline-levels 4
         :with-toc nil
         :section-numbers nil)))

;; Remove generated RST files so Sphinx reads pages derived from org sources.
(let ((rst-dir (expand-file-name "source" (file-name-directory (or load-file-name buffer-file-name)))))
  (dolist (rst-file (directory-files-recursively rst-dir "\\.rst$"))
    (delete-file rst-file)))

;; Run the publish
(org-publish "sphinx-rst" t)
