# Change log

## v0.7.3
- no upper Python version limit

## v0.7.1

- Add `flipud`, `fliplr` to `PIVImage`
- Bugfix example notebook

## v0.7.0

- Bugfix `rot90`,`rot180`

## v0.3.1

- PIVImagePair accepts None for the second filename. If so, the first image is cut in half as it is expected, that both
  images are within the first image.
- PIVImage got property `shape`, which returns the shape of the image.