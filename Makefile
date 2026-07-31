SHELL := /bin/sh

REGISTRY ?= ghcr.io
GITHUB_ORG ?= knows-aggregator
TAG ?= latest

# Add an image here only after images/<name>/Dockerfile is ready to build.
IMAGES := data-preparation model-training weight-aggregation

IMAGE_PREFIX := $(REGISTRY)/$(GITHUB_ORG)
SELECTED_IMAGES := $(if $(strip $(CONTAINER)),$(strip $(CONTAINER)),$(IMAGES))

.PHONY: help containers-list containers-validate containers-login containers-build containers-push

help:
	@echo "Container targets:"
	@echo "  make containers-list"
	@echo "  make containers-login GHCR_USER=<github-user>"
	@echo "  make containers-build [CONTAINER=<service>] [TAG=<tag>]"
	@echo "  make containers-push  [CONTAINER=<service>] [TAG=<tag>]"

containers-list:
	@for image in $(IMAGES); do \
		echo "$(IMAGE_PREFIX)/$$image:$(TAG)"; \
	done

containers-validate:
	@if [ -n "$(strip $(CONTAINER))" ]; then \
		case " $(IMAGES) " in \
			*" $(strip $(CONTAINER)) "*) ;; \
			*) echo "Unknown or unpublished image: $(strip $(CONTAINER))" >&2; \
			   echo "Available images: $(IMAGES)" >&2; \
			   exit 2 ;; \
		esac; \
	fi
	@for image in $(SELECTED_IMAGES); do \
		test -f "images/$$image/Dockerfile" || { \
			echo "Missing images/$$image/Dockerfile" >&2; \
			exit 2; \
		}; \
	done

containers-login:
	@test -n "$$GHCR_USER" || { echo "GHCR_USER is required" >&2; exit 2; }
	@test -n "$$CR_PAT" || { echo "CR_PAT is required" >&2; exit 2; }
	@printf '%s' "$$CR_PAT" | docker login "$(REGISTRY)" --username "$$GHCR_USER" --password-stdin

containers-build: containers-validate
	@set -eu; \
	for name in $(SELECTED_IMAGES); do \
		image="$(IMAGE_PREFIX)/$$name:$(TAG)"; \
		echo "Building $$image"; \
		docker build \
			--file "images/$$name/Dockerfile" \
			--tag "$$image" \
			.; \
	done

containers-push: containers-build
	@set -eu; \
	for name in $(SELECTED_IMAGES); do \
		image="$(IMAGE_PREFIX)/$$name:$(TAG)"; \
		echo "Pushing $$image"; \
		docker push "$$image"; \
	done
