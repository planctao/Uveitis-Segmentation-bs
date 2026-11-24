# Classification workspace

- Classification source, configuration, reports and experiment records belong under classification/.
- Record classification experiments in classification/EXPERIMENT_LOG.md. The repository-root EXPERIMENT_LOG.md records segmentation experiments; keep the two separate.
- threeclass/ preserves the current experiment source layout; legacy_binary/ preserves the earlier classification scripts and vendored-code license.
- Preserve frozen source and record bytes when they are referenced by model-package hashes. Make any changed implementation a new version and rerun its verification.
- Run python classification/tools/verify_snapshot.py after refreshing an archive. Check every exported file remains in Git and matches SNAPSHOT_MANIFEST.json.
- Keep data, patient/image manifests, individual predictions, weights, feature caches and raw training logs outside Git. Export aggregate metrics and protocol/audit summaries only.
- Keep development selection separate from external evaluation. Archiving or refreshing records does not authorize new training or new external testing.
