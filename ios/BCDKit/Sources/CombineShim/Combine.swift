// A stand-in for the two Combine names BCDKit uses, so the package also builds and tests
// on Linux (where Foundation ships without Combine). Only compiled there — see Package.swift.
// On Apple platforms the real Combine is imported and this module does not exist.
//
// `ScanCoordinator` is an `ObservableObject` with `@Published` state for SwiftUI's sake;
// nothing in the package subscribes to the publishers, so a wrapper that just stores the
// value is a faithful substitute for everything the tests exercise.
#if os(Linux)
public protocol ObservableObject: AnyObject {}

@propertyWrapper
public struct Published<Value> {
    public var wrappedValue: Value
    public init(wrappedValue: Value) { self.wrappedValue = wrappedValue }
}
#endif
